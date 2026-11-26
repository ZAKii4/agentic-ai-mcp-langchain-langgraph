"""Chatbot RAG agentique avec LangChain (create_agent) sur l'article « Attention Is All You Need ».

L'agent décide lui-même s'il faut chercher, avec quelles requêtes, et s'il faut recommencer.
Lancement : uv run python -m a_langchain.rag_chatbot
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common.llm import get_llm
from common.pdf_index import formater, rechercher

CONSIGNES = """You are a RAG assistant answering questions about the paper "Attention Is All You Need".
- For any question about the paper, call search_paper first, with a short ENGLISH query.
- If the question has several parts, search once per part. If a result is insufficient, rephrase and search again.
- Answer ONLY from the observations and cite pages like (p. 4). If the information is missing, say so.
- For greetings or unrelated questions, answer directly without searching.
- Always answer in French."""


@tool
def search_paper(query: str) -> str:
    """Search passages of the paper "Attention Is All You Need" (Transformer). Query in English."""
    return formater(rechercher(query))


def creer_chatbot():
    return create_agent(
        model=get_llm(),
        tools=[search_paper],
        system_prompt=CONSIGNES,
        middleware=[ToolCallLimitMiddleware(run_limit=5, exit_behavior="continue")],
        checkpointer=InMemorySaver(),
    )


def main():
    agent, config = creer_chatbot(), {"configurable": {"thread_id": "console"}}
    print("Chatbot RAG agentique. Tapez « quit » pour sortir.")
    while (question := input("\nVous : ").strip()).lower() not in {"quit", "exit", "q"}:
        if question:
            etat = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
            print("Assistant :", etat["messages"][-1].content)


if __name__ == "__main__":
    main()
