"""SAV multi-agents avec LangGraph.

START → orchestrateur ─documentation→ agent_documentaire (llm ⇄ tools : MCP rag_multimodal) ┐
              │      └─garantie─────→ agent_garantie     (llm ⇄ tools : consulter_garantie)  ┤
              │ ←────────────────────────────── retour ─────────────────────────────────────┘
              └─ réponse finale disponible → END

Lancer d'abord le serveur MCP (d_sav.mcp_server), puis :
uv run python -m d_sav.orchestrator "question" [--photo img_px200_01]
"""
import argparse
import asyncio
import json
import os
import re
from typing import Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel

from common.llm import get_llm
from d_sav.garantie import consulter_garantie, rediger_verdict

URL_MCP = os.getenv("SAV_MCP_URL", "http://127.0.0.1:8001/mcp")
MOTIF_PHOTO = re.compile(r"\bimg_[a-z0-9_]+")


class Entree(MessagesState):
    photo_ref: str


class Etat(Entree):
    question: str
    photo: str
    categorie: str
    reponse_finale: str


class Classification(BaseModel):
    categorie: Literal["documentation", "garantie"]


def orchestrateur(etat: Etat) -> Command[Literal["agent_documentaire", "agent_garantie", "__end__"]]:
    dernier = etat["messages"][-1]
    if not isinstance(dernier, HumanMessage):  # un agent a rendu la main
        if not etat.get("reponse_finale"):
            raise RuntimeError("Un agent a rendu la main sans réponse finale.")
        return Command(goto=END)
    question = dernier.text
    photo = etat.get("photo_ref") or next(iter(MOTIF_PHOTO.findall(question)), "")
    classe = get_llm().with_structured_output(Classification).invoke([
        SystemMessage("Classe la demande d'un client du service après-vente dans UNE seule catégorie :\n"
                      "- documentation : panne, voyant, pictogramme, utilisation ou dépannage d'un appareil ;\n"
                      "- garantie : savoir si un appareil est encore couvert, souvent avec un numéro de série."),
        HumanMessage(question + (f"\n[photo jointe : {photo}]" if photo else ""))])
    return Command(goto="agent_documentaire" if classe.categorie == "documentation" else "agent_garantie",
                   update={"question": question, "photo": photo, "categorie": classe.categorie,
                           "reponse_finale": "", "photo_ref": ""})


def messages_du_tour(etat: Etat) -> list:
    """Messages depuis la dernière question du client (une demande = un tour)."""
    debut = max(i for i, m in enumerate(etat["messages"]) if isinstance(m, HumanMessage))
    return etat["messages"][debut:]


def resultats_outil(etat: Etat, nom: str) -> list[dict]:
    resultats = []
    for m in messages_du_tour(etat):
        if isinstance(m, ToolMessage) and m.name == nom and m.status != "error":
            resultats.append(json.loads(m.text))
    return resultats


async def outils_mcp() -> list:
    client = MultiServerMCPClient({"sav": {"transport": "streamable_http", "url": URL_MCP}})
    return [o for o in await client.get_tools() if o.name == "rag_multimodal"]


# --- Agent documentaire -------------------------------------------------------------------------
async def llm_documentaire(etat: Etat):
    if any(isinstance(m, ToolMessage) for m in messages_du_tour(etat)):
        return {"messages": [AIMessage("Résultat documentaire reçu.")]}
    # La réponse doit être sourcée : l'appel à l'outil est obligatoire, avec la question et la photo
    # d'origine (le LLM ne peut ni les reformuler ni les oublier).
    appel = {"name": "rag_multimodal", "id": f"doc_{uuid4().hex}",
             "args": {"question": etat["question"], "photo_ref": etat["photo"]}}
    return {"messages": [AIMessage(content="", tool_calls=[appel])]}


async def tools_documentaire(etat: Etat):
    try:
        return await ToolNode(await outils_mcp()).ainvoke({"messages": etat["messages"]})
    except Exception as erreur:  # serveur MCP arrêté, délai dépassé…
        appel = etat["messages"][-1].tool_calls[0]
        return {"messages": [ToolMessage(f"Serveur MCP indisponible : {type(erreur).__name__}", status="error",
                                         name=appel["name"], tool_call_id=appel["id"])]}


def conclure_documentaire(etat: Etat):
    resultats = resultats_outil(etat, "rag_multimodal")
    if not resultats:
        texte = "Le service documentaire est momentanément indisponible. Réessayez plus tard."
    else:
        r = resultats[-1]
        lignes = [f"- {s['manuel']}, p. {s['page']}" + (f" ({s['figure']})" if s["type"] == "figure" else "")
                  for s in r["sources"]]
        titre = "Pages consultées (aucune citée)" if any(s["type"] == "consulte" for s in r["sources"]) else "Sources"
        texte = r["reponse"] + (f"\n\n{titre} :\n" + "\n".join(lignes) if lignes else "")
    return {"reponse_finale": texte, "messages": [AIMessage(texte, name="agent_documentaire")]}


# --- Agent garantie ------------------------------------------------------------------------------
def llm_garantie(etat: Etat):
    llm = get_llm().bind_tools([consulter_garantie])
    consigne = SystemMessage(
        "Tu es l'agent garantie. Appelle consulter_garantie avec le numéro de série donné par le client. "
        "N'invente jamais de numéro : s'il manque, demande-le en français, sans appeler l'outil.")
    return {"messages": [llm.invoke([consigne, *messages_du_tour(etat)])]}


def conclure_garantie(etat: Etat):
    saisie = etat["question"].upper()
    # On n'accepte que les numéros réellement écrits par le client.
    resultats = [r for r in resultats_outil(etat, "consulter_garantie") if r["numero_serie"] in saisie]
    if resultats:
        uniques = {r["numero_serie"]: r for r in resultats}
        texte = "\n".join(rediger_verdict(r) for r in uniques.values())
    else:
        texte = "Pour vérifier la garantie, indiquez le numéro de série de l'appareil (exemple : PX200-0042)."
    return {"reponse_finale": texte, "messages": [AIMessage(texte, name="agent_garantie")]}


def sous_agent(llm, tools, conclure):
    """Boucle ReAct d'un agent spécialisé : llm ⇄ tools, puis conclure."""
    g = StateGraph(Etat)
    g.add_node("llm", llm)
    g.add_node("tools", tools)
    g.add_node("conclure", conclure)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: "conclure"})
    g.add_edge("tools", "llm")
    g.add_edge("conclure", END)
    return g.compile()


def construire_graphe(checkpointer=None):
    g = StateGraph(Etat, input_schema=Entree)
    g.add_node("orchestrateur", orchestrateur)
    g.add_node("agent_documentaire", sous_agent(llm_documentaire, tools_documentaire, conclure_documentaire))
    g.add_node("agent_garantie", sous_agent(llm_garantie, ToolNode([consulter_garantie]), conclure_garantie))
    g.add_edge(START, "orchestrateur")
    g.add_edge("agent_documentaire", "orchestrateur")
    g.add_edge("agent_garantie", "orchestrateur")
    return g.compile(checkpointer=checkpointer)


graph = construire_graphe()  # chargé par LangGraph Studio


async def main():
    arguments = argparse.ArgumentParser(description="SAV multi-agents")
    arguments.add_argument("question")
    arguments.add_argument("--photo", default="", help="référence de la photo, ex. img_px200_01")
    a = arguments.parse_args()
    entree = {"messages": [HumanMessage(a.question)], "photo_ref": a.photo}
    final = {}
    async for chemin, mode, donnees in graph.astream(entree, {"recursion_limit": 25},
                                                     stream_mode=["updates", "values"], subgraphs=True):
        if mode == "updates":
            for noeud in donnees:
                print("  nœud :", (chemin[0].split(":")[0] + " / " if chemin else "") + noeud)
        elif not chemin:
            final = donnees
    print(f"\nCatégorie : {final['categorie']}\n{final['reponse_finale']}")


if __name__ == "__main__":
    asyncio.run(main())
