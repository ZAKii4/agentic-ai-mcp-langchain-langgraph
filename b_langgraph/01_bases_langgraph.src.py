# %% [markdown]
# # Partie B.1 : les bases de LangGraph
# LangGraph décrit un agent comme un **graphe** : un **état** partagé, des **nœuds** (fonctions) qui le
# mettent à jour, et des **arêtes** qui décident du nœud suivant.

# %%
import sys, operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

RACINE = Path.cwd().parent if Path.cwd().name == "b_langgraph" else Path.cwd()
sys.path.insert(0, str(RACINE))
from langgraph.graph import StateGraph, MessagesState, START, END

def dessiner(g):
    print(g.get_graph().draw_mermaid())

# %% [markdown]
# ## 1. État, nœuds, arêtes
# Chaque nœud reçoit l'état et renvoie une **mise à jour partielle**.

# %%
class Etat(TypedDict):
    texte: str

def majuscules(e: Etat):
    return {"texte": e["texte"].upper()}

def exclamer(e: Etat):
    return {"texte": e["texte"] + " !"}

b = StateGraph(Etat)
b.add_node("majuscules", majuscules)
b.add_node("exclamer", exclamer)
b.add_edge(START, "majuscules")
b.add_edge("majuscules", "exclamer")
b.add_edge("exclamer", END)
g = b.compile()
print(g.invoke({"texte": "bonjour langgraph"}))
dessiner(g)

# %% [markdown]
# ## 2. Reducers
# Par défaut une mise à jour **remplace** la valeur. Un reducer (`Annotated[type, fonction]`) la **combine**.

# %%
class EtatJournal(TypedDict):
    journal: Annotated[list[str], operator.add]
    dernier: str

b = StateGraph(EtatJournal)
b.add_node("a", lambda e: {"journal": ["A"], "dernier": "A"})
b.add_node("b", lambda e: {"journal": ["B"], "dernier": "B"})
b.add_edge(START, "a"); b.add_edge("a", "b"); b.add_edge("b", END)
print(b.compile().invoke({"journal": ["début"], "dernier": ""}))

# %% [markdown]
# ## 3. Arêtes conditionnelles et boucle

# %%
class Compteur(TypedDict):
    n: int

def continuer(e: Compteur) -> Literal["incrementer", "__end__"]:
    return "incrementer" if e["n"] < 5 else END

b = StateGraph(Compteur)
b.add_node("incrementer", lambda e: {"n": e["n"] + 1})
b.add_edge(START, "incrementer")
b.add_conditional_edges("incrementer", continuer)
g = b.compile()
print(g.invoke({"n": 0}, {"recursion_limit": 10}))
dessiner(g)

# %% [markdown]
# ## 4. `Command` : mettre à jour l'état **et** router depuis le nœud

# %%
from langgraph.types import Command, Send, interrupt

class Demande(TypedDict):
    nombre: int
    resultat: str

def trier(e: Demande) -> Command[Literal["pair", "impair"]]:
    return Command(goto="pair" if e["nombre"] % 2 == 0 else "impair")

b = StateGraph(Demande)
b.add_node("trier", trier)
b.add_node("pair", lambda e: {"resultat": "pair"})
b.add_node("impair", lambda e: {"resultat": "impair"})
b.add_edge(START, "trier"); b.add_edge("pair", END); b.add_edge("impair", END)
g = b.compile()
print(g.invoke({"nombre": 7}), g.invoke({"nombre": 10}))

# %% [markdown]
# ## 5. Parallélisme dynamique avec `Send` (map-reduce)

# %%
class Lot(TypedDict):
    mots: list[str]
    longueurs: Annotated[list[tuple[str, int]], operator.add]

b = StateGraph(Lot)
b.add_node("mesurer", lambda tache: {"longueurs": [(tache["mot"], len(tache["mot"]))]})
b.add_conditional_edges(START, lambda e: [Send("mesurer", {"mot": m}) for m in e["mots"]], ["mesurer"])
b.add_edge("mesurer", END)
print(b.compile().invoke({"mots": ["agent", "graphe", "état"]}))

# %% [markdown]
# ## 6. Un chatbot : `MessagesState`

# %%
from langchain_core.messages import HumanMessage, SystemMessage
from common.llm import get_llm

llm = get_llm()

def chatbot(e: MessagesState):
    return {"messages": [llm.invoke([SystemMessage("Réponds en français, en une phrase."), *e["messages"]])]}

b = StateGraph(MessagesState)
b.add_node("chatbot", chatbot)
b.add_edge(START, "chatbot")
print(b.compile().invoke({"messages": [HumanMessage("Qu'est-ce qu'un graphe ?")]})["messages"][-1].content)

# %% [markdown]
# ## 7. Agent ReAct construit à la main : `ToolNode` et `tools_condition`

# %%
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def multiplier(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def additionner(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

outils = [multiplier, additionner]
llm_outils = llm.bind_tools(outils)

def agent(e: MessagesState):
    consigne = SystemMessage("Use the tools for every calculation, one step at a time. Answer in French.")
    return {"messages": [llm_outils.invoke([consigne, *e["messages"]])]}

b = StateGraph(MessagesState)
b.add_node("agent", agent)
b.add_node("tools", ToolNode(outils))
b.add_edge(START, "agent")
b.add_conditional_edges("agent", tools_condition)
b.add_edge("tools", "agent")
react = b.compile()
dessiner(react)
for m in react.invoke({"messages": [HumanMessage("Calcule 23 × 17 puis ajoute 9.")]})["messages"]:
    m.pretty_print()

# %% [markdown]
# ## 8. Persistance : checkpointer, `thread_id`, historique et retour dans le temps

# %%
from langgraph.checkpoint.memory import InMemorySaver

memo = b.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}
memo.invoke({"messages": [HumanMessage("Je m'appelle Mohamed. Réponds juste « ok ».")]}, config)
print(memo.invoke({"messages": [HumanMessage("Quel est mon prénom ?")]}, config)["messages"][-1].content)

historique = list(memo.get_state_history(config))
print(len(historique), "checkpoints ; état courant :", len(memo.get_state(config).values["messages"]), "messages")
fin_tour_1 = [s for s in historique if not s.next and len(s.values["messages"]) == 2][0]
print("Retour à la fin du tour 1 :", [m.content for m in fin_tour_1.values["messages"]])

# %% [markdown]
# ## 9. Human-in-the-loop avec `interrupt`

# %%
class Remboursement(TypedDict):
    montant: float
    statut: str

def validation(e: Remboursement):
    accord = interrupt({"question": f"Approuver un remboursement de {e['montant']} DH ?"})
    return {"statut": "approuvé" if accord == "oui" else "refusé"}

b = StateGraph(Remboursement)
b.add_node("validation", validation)
b.add_edge(START, "validation")
g = b.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "r1"}}
pause = g.invoke({"montant": 500, "statut": ""}, config)
print(pause["__interrupt__"][0].value)
print(g.invoke(Command(resume="oui"), config))

# %% [markdown]
# ## 10. Streaming

# %%
for mode in ("updates", "values"):
    print(f"--- stream_mode={mode}")
    for etape in react.stream({"messages": [HumanMessage("Combien font 6 × 7 ?")]}, stream_mode=mode):
        if mode == "updates":
            print("nœud exécuté :", list(etape))
        else:
            print("messages dans l'état :", len(etape["messages"]))
print("--- stream_mode=messages (token par token)")
for fragment, meta in react.stream({"messages": [HumanMessage("Dis bonjour.")]}, stream_mode="messages"):
    print(fragment.content, end="|" if fragment.content else "")

# %% [markdown]
# ## 11. Sous-graphes
# Un graphe compilé peut être utilisé comme nœud d'un autre. C'est le principe des systèmes multi-agents.

# %%
sous = StateGraph(Etat)
sous.add_node("majuscules", majuscules); sous.add_node("exclamer", exclamer)
sous.add_edge(START, "majuscules"); sous.add_edge("majuscules", "exclamer")

parent = StateGraph(Etat)
parent.add_node("nettoyer", lambda e: {"texte": " ".join(e["texte"].split())})
parent.add_node("mise_en_forme", sous.compile())
parent.add_edge(START, "nettoyer"); parent.add_edge("nettoyer", "mise_en_forme")
print(parent.compile().invoke({"texte": "  les    sous-graphes  "}))
