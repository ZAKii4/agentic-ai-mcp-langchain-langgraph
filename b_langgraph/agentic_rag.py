"""Chatbot Agentic RAG avec LangGraph sur « Attention Is All You Need ».

START → preparer → agent ─(réponse directe)→ END
                     └─(appel d'outil)→ rechercher → évaluer ─pertinent→ generer → END
                                           ↑                  └─non pertinent→ reformuler ┘
La reformulation est limitée à MAX_REFORMULATIONS pour garantir l'arrêt.
"""
from typing import Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from common.llm import get_llm
from common.pdf_index import formater, rechercher

MAX_REFORMULATIONS = 2

CONSIGNES_AGENT = """You are the router of a RAG assistant about the paper "Attention Is All You Need".
- For any question about the paper, the Transformer or attention, call search_paper with a short ENGLISH query.
  Resolve references using the conversation and keep precise qualifiers (e.g. "base model", language pair).
  For a multi-part question, call the tool once per part.
- A question that mentions "the article" or "the paper" must be searched, even if you expect no answer.
- For greetings or unrelated questions, answer directly in French without any tool."""


class Etat(MessagesState):
    question: str
    reformulations: int
    pertinent: bool


class Evaluation(BaseModel):
    pertinent: bool = Field(description="True if the passages contain information that helps answer the question")


@tool
def search_paper(query: str) -> str:
    """Search passages of the paper "Attention Is All You Need". The query must be in English."""
    return formater(rechercher(query))


def observations(etat: Etat) -> str:
    """Résultats d'outils produits depuis la dernière recherche lancée."""
    blocs = []
    for message in reversed(etat["messages"]):
        if isinstance(message, ToolMessage):
            blocs.append(str(message.content))
        elif blocs:
            break
    return "\n\n".join(reversed(blocs))


def preparer(etat: Etat):
    question = next(m for m in reversed(etat["messages"]) if isinstance(m, HumanMessage))
    return {"question": question.text, "reformulations": 0, "pertinent": False}


def contexte_conversation(etat: Etat) -> str:
    """Questions précédentes, pour résoudre « il », « ce modèle », etc."""
    precedentes = [m.text for m in etat["messages"] if isinstance(m, HumanMessage)][-4:-1]
    return "\n".join(f"- {q}" for q in precedentes) or "(none)"


def agent(etat: Etat):
    llm = get_llm().bind_tools([search_paper])
    return {"messages": [llm.invoke([SystemMessage(CONSIGNES_AGENT), *etat["messages"]])]}


def evaluer(etat: Etat):
    juge = get_llm().with_structured_output(Evaluation)
    requetes = [a["args"].get("query", "") for m in etat["messages"] if isinstance(m, AIMessage)
                for a in m.tool_calls][-3:]
    verdict = juge.invoke(
        "Grade the relevance of retrieved passages. Be lenient: relevant if they contain facts that help answer.\n"
        f"Question: {etat['question']}\nSearch queries: {requetes}\nPassages:\n{observations(etat)}")
    return {"pertinent": verdict.pertinent}


def apres_evaluation(etat: Etat) -> Literal["generer", "reformuler"]:
    if etat["pertinent"] or etat["reformulations"] >= MAX_REFORMULATIONS:
        return "generer"
    return "reformuler"


def reformuler(etat: Etat):
    requete = get_llm().invoke(
        "Rewrite this question as one short ENGLISH search query for the paper 'Attention Is All You Need', "
        f"using different technical keywords. Output the query only.\n"
        f"Previous questions:\n{contexte_conversation(etat)}\nQuestion: {etat['question']}").text.strip()
    appel = {"name": search_paper.name, "args": {"query": requete}, "id": f"reformulation_{uuid4().hex}"}
    return {"messages": [AIMessage(content="", tool_calls=[appel])],
            "reformulations": etat["reformulations"] + 1}


def generer(etat: Etat):
    reserve = "" if etat["pertinent"] else (
        "\nThe retrieved passages were judged insufficient: say clearly that the information was not found.")
    reponse = get_llm().invoke([
        SystemMessage("Answer in French using ONLY the passages. Cite pages like (p. 4). "
                      "If the information is missing, say so. Be concise.\n"
        "The paper reports several model variants (base and big). Use the previous questions to resolve "
        "references. In a results table, read the row of the variant being discussed; "
        "a summary sentence may refer to another variant." + reserve),
        HumanMessage(f"Previous questions:\n{contexte_conversation(etat)}\n\n"
                     f"Current question: {etat['question']}\n\nPassages:\n{observations(etat)}")])
    return {"messages": [reponse]}


def construire_graphe(checkpointer=None):
    g = StateGraph(Etat, input_schema=MessagesState)
    g.add_node("preparer", preparer)
    g.add_node("agent", agent)
    g.add_node("rechercher", ToolNode([search_paper]))
    g.add_node("evaluer", evaluer)
    g.add_node("reformuler", reformuler)
    g.add_node("generer", generer)
    g.add_edge(START, "preparer")
    g.add_edge("preparer", "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "rechercher", END: END})
    g.add_edge("rechercher", "evaluer")
    g.add_conditional_edges("evaluer", apres_evaluation)
    g.add_edge("reformuler", "rechercher")
    g.add_edge("generer", END)
    return g.compile(checkpointer=checkpointer)


# Graphe chargé par LangGraph Studio (la plateforme fournit elle-même la persistance)
graph = construire_graphe()
