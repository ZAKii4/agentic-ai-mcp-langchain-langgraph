"""Interface web du chatbot Agentic RAG.  Lancement : uv run streamlit run b_langgraph/app_streamlit.py"""
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from b_langgraph.agentic_rag import construire_graphe
from common.llm import PROVIDER, nom_modele
from common.pdf_index import indexer

st.set_page_config(page_title="Agentic RAG", page_icon="📚")
st.title("📚 Chatbot Agentic RAG (LangGraph)")
st.caption("Document : « Attention Is All You Need »")


@st.cache_resource(show_spinner="Indexation du document…")
def charger_graphe():
    indexer()
    return construire_graphe(InMemorySaver())


def nouvelle_conversation():
    st.session_state.thread_id = str(uuid4())
    st.session_state.historique = []


if "thread_id" not in st.session_state:
    nouvelle_conversation()

with st.sidebar:
    st.write(f"**Modèle** : {PROVIDER} / {nom_modele()}")
    st.button("Nouvelle conversation", on_click=nouvelle_conversation)

for tour in st.session_state.historique:
    with st.chat_message(tour["role"]):
        if tour.get("etapes"):
            with st.expander("Étapes de l'agent"):
                st.markdown("\n".join(f"- {e}" for e in tour["etapes"]))
        st.markdown(tour["texte"])


def decrire(noeud, maj):
    if noeud in ("agent", "reformuler"):
        for message in maj.get("messages", []):
            for appel in getattr(message, "tool_calls", []):
                prefixe = "🔁 Reformulation" if noeud == "reformuler" else "🔎 Recherche"
                yield f"{prefixe} : `{appel['args'].get('query', '')}`"
    elif noeud == "rechercher":
        yield f"📄 {sum(isinstance(m, ToolMessage) for m in maj['messages'])} résultat(s) de recherche reçu(s)"
    elif noeud == "evaluer":
        yield "✅ Passages jugés pertinents" if maj["pertinent"] else "❌ Passages jugés insuffisants"


if question := st.chat_input("Posez votre question sur l'article…"):
    st.session_state.historique.append({"role": "user", "texte": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        etapes, reponse = [], ""
        try:
            graphe = charger_graphe()
            with st.status("L'agent travaille…") as statut:
                config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 20}
                for evenement in graphe.stream({"messages": [HumanMessage(question)]}, config,
                                               stream_mode="updates"):
                    for noeud, maj in evenement.items():
                        for etape in decrire(noeud, maj or {}):
                            etapes.append(etape)
                            st.write(etape)
                        if noeud in ("agent", "generer"):
                            for message in maj["messages"]:
                                if isinstance(message, AIMessage) and not message.tool_calls:
                                    reponse = message.text
                statut.update(label="Terminé", state="complete")
            st.markdown(reponse)
        except Exception as erreur:
            reponse = f"Erreur : {erreur}. Vérifiez qu'Ollama est lancé (ou la clé OpenAI)."
            st.error(reponse)
        st.session_state.historique.append({"role": "assistant", "texte": reponse, "etapes": etapes})
