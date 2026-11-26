"""Agent LangChain qui utilise le serveur MCP via streamable-http.

Lancer d'abord le serveur, puis : uv run python -m c_mcp.agent_client ["votre question"]
"""
import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.llm import get_llm

URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
QUESTION = "Quel est le salaire de Mohamed, et quel temps fait-il à Marrakech ?"


async def main(question: str):
    client = MultiServerMCPClient({"demo": {"transport": "streamable_http", "url": URL}})
    outils = await client.get_tools()
    print("Tools découverts :", [o.name for o in outils])
    agent = create_agent(
        get_llm(), outils,
        system_prompt="Use the tools to answer; never invent data, units or currencies. If a tool fails, say so. Answer in French.")
    etat = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    for message in etat["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for appel in message.tool_calls:
                print(f"→ ACTION {appel['name']} {appel['args']}")
        elif isinstance(message, ToolMessage):
            print(f"← OBSERVATION {message.text}")
    print("Réponse :", etat["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or QUESTION))
