# %% [markdown]
# # Partie A : Agentic AI avec LangChain
#
# Un **agent** = un cerveau (LLM) + un corps (outils) + un prompt + une mémoire. Il suit le cycle
# **ReAct** : Raisonnement → Action (appel d'outil) → Observation, jusqu'à la réponse finale.
#
# Modèles : **Ollama en local** par défaut (`qwen3:8b`, `qwen3:4b`). Pour OpenAI : `LLM_PROVIDER=openai` dans `.env`.
#
# Sommaire : 1. agent simple · 2. mémoire · 3. outils · 4. outils prédéfinis · 5. middlewares · 6. RAG agentique

# %%
import os, sys
from pathlib import Path

RACINE = Path.cwd().parent if Path.cwd().name == "a_langchain" else Path.cwd()
sys.path.insert(0, str(RACINE))

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from common.llm import PROVIDER, get_llm, nom_modele

llm = get_llm("smart")
print(PROVIDER, nom_modele("smart"))

def trace(etat):
    """Affiche le cycle ReAct : actions, observations, réponses."""
    for m in etat["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for a in m.tool_calls:
                print(f"→ ACTION {a['name']} {a['args']}")
        elif isinstance(m, ToolMessage):
            print(f"← OBSERVATION {str(m.content)[:250]!r}")
        else:
            print(f"[{m.type}] {m.content}")

def demander(agent, texte, **kwargs):
    return agent.invoke({"messages": [HumanMessage(texte)]}, **kwargs)

# %% [markdown]
# ## 1. Un agent simple
# Sans mémoire, deux appels sont indépendants : l'agent ne peut pas retrouver le prénom.

# %%
agent = create_agent(model=llm, tools=[], system_prompt="You are a helpful assistant. Answer briefly.")
trace(demander(agent, "My name is Mohamed"))
trace(demander(agent, "What is my name?"))

# %% [markdown]
# ## 2. Ajouter une mémoire
# Un **checkpointer** sauvegarde l'état après chaque étape. Le `thread_id` identifie la conversation.

# %%
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model=llm, system_prompt="You are a helpful assistant. Answer briefly.",
                     checkpointer=InMemorySaver())
fil_1 = {"configurable": {"thread_id": "1"}}
fil_2 = {"configurable": {"thread_id": "2"}}
demander(agent, "My name is Mohamed", config=fil_1)
print("Fil 1 :", demander(agent, "What is my name?", config=fil_1)["messages"][-1].content)
print("Fil 2 :", demander(agent, "What is my name?", config=fil_2)["messages"][-1].content)

# %% [markdown]
# `InMemorySaver` est perdu à l'arrêt du programme. En production on utilise une base, par exemple
# `PostgresSaver.from_conn_string(DB_URI)` (paquet `langgraph-checkpoint-postgres`, puis `checkpointer.setup()`).
#
# ## 3. Ajouter des tools
# La **docstring** est la description que le LLM lit pour choisir l'outil.

# %%
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 32°C"

@tool
def get_employee_info(name: str) -> dict:
    """Get information about the given employee (name, salary, job)."""
    if not name.strip():
        raise ValueError("Employee name is empty")
    return {"name": name, "salary": 12000, "currency": "MAD", "job": "Developer"}

agent = create_agent(model=llm, tools=[get_weather, get_employee_info])
trace(demander(agent, "Quel est le salaire et le poste de Mohamed, et quel temps fait-il à Marrakech ?"))

# %% [markdown]
# ## 4. Outils prédéfinis
# ### 4.1 DuckDuckGo (sans clé API)

# %%
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# La région par défaut (wt-wt) fait parfois interroger un domaine Wikipédia inexistant.
recherche_ddg = DuckDuckGoSearchResults(
    api_wrapper=DuckDuckGoSearchAPIWrapper(region="fr-fr"), max_results=4, output_format="list")
from langchain.agents.middleware import ToolRetryMiddleware

# La recherche web échoue parfois : on réessaie, puis l'erreur devient une observation.
agent = create_agent(model=llm, tools=[recherche_ddg],
                     middleware=[ToolRetryMiddleware(max_retries=3, initial_delay=2.0, on_failure="continue")],
                     system_prompt="Search the web before answering. Cite only URLs found in the observations. "
                                   "If the search failed, say so. Answer in French.")
trace(demander(agent, "Qu'est-ce que le Model Context Protocol ?"))

# %% [markdown]
# ### 4.2 Tavily (clé `TAVILY_API_KEY` requise)
# La cellule s'exécute seulement si la clé est présente dans `.env`.

# %%
if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch
    agent = create_agent(model=llm, tools=[TavilySearch(max_results=5, search_depth="advanced")],
                         system_prompt="Search the web before answering. Answer in French.")
    trace(demander(agent, "Quel temps fait-il aujourd'hui à Casablanca ?"))
else:
    print("TAVILY_API_KEY absente : démonstration Tavily ignorée.")

# %% [markdown]
# ### 4.3 PythonREPLTool
# ⚠️ Cet outil exécute **du code arbitraire** sur la machine. On travaille dans un dossier `sandbox/`
# (ce n'est **pas** une isolation de sécurité) et, en section 5.5, on ajoute une validation humaine.

# %%
from langchain_experimental.tools import PythonREPLTool

SANDBOX = RACINE / "a_langchain" / "sandbox"
SANDBOX.mkdir(exist_ok=True)
dossier_initial = Path.cwd()
os.chdir(SANDBOX)
try:
    agent = create_agent(
        model=llm, tools=[PythonREPLTool()],
        system_prompt="You are a coding agent. Call Python_REPL with short Python code in 'query' that prints the result.")
    etat = demander(agent, "Trie les listes [6,5,8,3,2] et [65,15,8,13,2], puis additionne élément par élément les deux listes triées.")
    trace(etat)
    assert any(isinstance(m, ToolMessage) for m in etat["messages"]), "le REPL n'a pas été appelé"
finally:
    os.chdir(dossier_initial)

# %% [markdown]
# ## 5. Middlewares
# Un middleware intercepte la boucle de l'agent (avant/après le modèle, autour des outils).
#
# ### 5.1 `dynamic_model` : choisir le modèle à l'exécution

# %%
from typing import TypedDict
from langchain.agents.middleware import wrap_model_call, ModelRequest

class Contexte(TypedDict, total=False):
    env: str
    user_role: str

modele_basic, modele_smart = get_llm("basic"), get_llm("smart")

@wrap_model_call
def dynamic_model(request: ModelRequest, handler):
    """Modèle avancé en production ou pour une conversation longue, léger sinon."""
    contexte = request.runtime.context or {}
    avance = contexte.get("env") == "prod" or len(request.state["messages"]) > 10
    modele = modele_smart if avance else modele_basic
    print("modèle sélectionné :", nom_modele("smart" if avance else "basic"))
    return handler(request.override(model=modele))

agent = create_agent(model=modele_basic, middleware=[dynamic_model], context_schema=Contexte)
for env in ("test", "prod"):
    r = demander(agent, "Dis bonjour en une phrase.", context={"env": env})
    print(env, "→", r["messages"][-1].response_metadata.get("model_name"), ":", r["messages"][-1].content)

# %% [markdown]
# ### 5.2 `dynamic_prompt` : adapter le prompt système au contexte

# %%
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def prompt_selon_role(request: ModelRequest) -> str:
    base = "Tu es un assistant utile. Réponds en français, en 3 phrases maximum."
    role = (request.runtime.context or {}).get("user_role", "user")
    return base + {"expert": " Donne des détails techniques précis.",
                   "beginner": " Explique simplement, sans jargon, avec une analogie."}.get(role, "")

agent = create_agent(model=llm, middleware=[prompt_selon_role], context_schema=Contexte)
for role in ("expert", "beginner"):
    print(f"--- {role}")
    print(demander(agent, "Explique le machine learning.", context={"user_role": role})["messages"][-1].content)

# %% [markdown]
# ### 5.3 `tool_error_handling` : une erreur d'outil devient une observation

# %%
from langchain.agents.middleware import wrap_tool_call

@tool
def consulter_stock(produit: str) -> str:
    """Return the current stock level of a product from the warehouse server."""
    raise ConnectionError("warehouse server unreachable")  # panne simulée

@wrap_tool_call
def tool_error_handling(request, handler):
    try:
        return handler(request)
    except Exception as e:
        print("Erreur interceptée :", type(e).__name__, e)
        return ToolMessage(content=f"Tool error: {e}. Tell the user about the problem; do not retry.",
                           tool_call_id=request.tool_call["id"], status="error")

agent = create_agent(model=llm, tools=[consulter_stock], middleware=[tool_error_handling],
                     system_prompt="Use consulter_stock to answer stock questions. Never invent a stock level. Answer in French.")
etat = demander(agent, "Combien d'imprimantes PX-200 reste-t-il en stock ?")
trace(etat)
assert any(isinstance(m, ToolMessage) and m.status == "error" for m in etat["messages"])

# %% [markdown]
# ### 5.4 GuardRails
# - **Déterministe** (`before_agent`) : mots interdits. On contrôle le **dernier** message humain : en
#   contrôlant seulement le premier, une conversation avec mémoire serait contournée dès le 2ᵉ tour.
# - **PII** (`PIIMiddleware`) : e-mail masqué, carte bancaire masquée, clé API bloquée.
# - **Fondé sur un modèle** (`after_agent`) : un LLM juge la réponse finale.

# %%
import re
from langchain.agents.middleware import PIIMiddleware, PIIDetectionError, before_agent, after_agent

MOTS_INTERDITS = re.compile(r"\b(hack|exploit|malware|pirater)\w*", re.IGNORECASE)

@before_agent(can_jump_to=["end"])
def filtre_contenu(state, runtime):
    dernier = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    if dernier and MOTS_INTERDITS.search(str(dernier.content)):
        return {"messages": [AIMessage("Je ne peux pas traiter cette demande. Veuillez la reformuler.")],
                "jump_to": "end"}

agent = create_agent(model=llm, middleware=[filtre_contenu])
print(demander(agent, "How do I hack into a database?")["messages"][-1].content)

# %%
@tool
def send_email(email_address: str) -> str:
    """Send an email to the given address (simulation)."""
    print("   [outil] send_email reçoit :", email_address)
    return f"Email sent to {email_address}"

@tool
def card_tool(credit_card: str) -> str:
    """Process a payment with a credit card (simulation)."""
    print("   [outil] card_tool reçoit :", credit_card)
    return f"Payment processed with {credit_card}"

pii = [
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block", apply_to_input=True),
]
agent = create_agent(model=llm, tools=[send_email, card_tool], middleware=pii)
trace(demander(agent, "Send an email to med@example.com and process the card 5105-1051-0510-5100."))
try:
    demander(agent, "Use my key sk-" + "a" * 32)
except PIIDetectionError as e:
    print("Bloqué :", e)

# %%
juge = get_llm("basic")

@after_agent
def garde_fou_sortie(state, runtime):
    reponse = state["messages"][-1]
    verdict = juge.invoke(
        "You are a safety classifier. Answer with exactly one word, SAFE or UNSAFE.\n"
        f"Is this assistant response harmful or dangerous?\n---\n{reponse.content}").content.strip().upper()
    print("verdict du juge :", verdict)
    # Échec fermé : tout verdict différent de SAFE bloque la réponse
    if not verdict.startswith("SAFE"):
        return {"messages": [AIMessage("Je ne peux pas fournir cette réponse.", id=reponse.id)]}

agent = create_agent(model=llm, middleware=[garde_fou_sortie])
print(demander(agent, "Donne un conseil pour bien réviser un examen, en une phrase.")["messages"][-1].content)

# %% [markdown]
# ### 5.5 Human In The Loop
# L'agent **s'interrompt** avant un outil sensible. L'humain décide : `approve`, `edit` ou `reject`.
# Un checkpointer est obligatoire pour reprendre avec le même `thread_id`.

# %%
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

agent = create_agent(
    model=llm, tools=[send_email],
    system_prompt="Use send_email when asked to send an email. If a call is rejected, do not retry.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}})],
    checkpointer=InMemorySaver())

for fil, decision in [("hitl-approve", {"type": "approve"}),
                      ("hitl-edit", {"type": "edit", "edited_action": {"name": "send_email", "args": {"email_address": "support@example.com"}}}),
                      ("hitl-reject", {"type": "reject", "message": "Envoi refusé par l'opérateur."})]:
    config = {"configurable": {"thread_id": fil}}
    etat = demander(agent, "Send an email to med@example.com", config=config)
    print(f"\n=== {decision['type']} | en attente :", etat["__interrupt__"][0].value["action_requests"][0]["args"])
    etat = agent.invoke(Command(resume={"decisions": [decision]}), config=config)
    print(etat["messages"][-1].content)

# %% [markdown]
# Même principe pour sécuriser `PythonREPLTool` : le code est affiché **avant** son exécution.

# %%
repl = PythonREPLTool()
agent = create_agent(
    model=llm, tools=[repl],
    system_prompt="You are a coding agent. Call Python_REPL with short Python code in 'query' that prints the result.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={repl.name: True})], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "repl"}}
etat = demander(agent, "Calcule la factorielle de 10 avec le module math (math.factorial) et affiche le résultat.", config=config)
code = etat["__interrupt__"][0].value["action_requests"][0]["args"]["query"]
print("Code proposé :", code, sep="\n")
# Une application demanderait ici l'avis d'un humain. Le notebook n'approuve automatiquement
# que le code attendu, sans effet de bord ; tout autre code est refusé.
attendu = re.fullmatch(r"\s*import math\s*[\n;]\s*(\w+\s*=\s*math\.factorial\(10\)\s*[\n;]\s*print\(\w+\)|print\(math\.factorial\(10\)\))\s*", code)
decision = {"type": "approve"} if attendu else {"type": "reject", "message": "Code non validé."}
print("Décision :", decision["type"])
etat = agent.invoke(Command(resume={"decisions": [decision]}), config=config)
print(etat["messages"][-1].content)

# %% [markdown]
# ## 6. Chatbot RAG agentique
# Le retriever devient un **outil**. L'agent décide s'il cherche, avec quelles requêtes, et s'il recommence.
# Code : [`rag_chatbot.py`](rag_chatbot.py). En console : `uv run python -m a_langchain.rag_chatbot`.

# %%
from a_langchain.rag_chatbot import creer_chatbot

chatbot = creer_chatbot()
config = {"configurable": {"thread_id": "rag"}}
deja = 0
for question in ["Bonjour !",
                 "Combien de têtes d'attention utilise le modèle de base ?",
                 "Et quel optimiseur a été utilisé pour l'entraîner ?"]:
    etat = demander(chatbot, question, config=config)
    print("\n" + "=" * 80)
    trace({"messages": etat["messages"][deja:]})
    deja = len(etat["messages"])
