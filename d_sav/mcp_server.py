"""Serveur MCP du SAV : publie l'outil distant rag_multimodal (streamable-http).

Lancement : uv run python -m d_sav.mcp_server      → http://127.0.0.1:8001/mcp
"""
import asyncio
import os

from mcp.server.fastmcp import FastMCP

from d_sav.rag.agent_rag import repondre

mcp = FastMCP("sav-documentation", host=os.getenv("SAV_MCP_HOST", "127.0.0.1"),
              port=int(os.getenv("SAV_MCP_PORT", "8001")))


@mcp.tool()
async def rag_multimodal(question: str, photo_ref: str = "") -> dict:
    """Answer a troubleshooting question from the product manuals (text and figures), with sources.

    Args:
        question: the customer's question, in French.
        photo_ref: optional reference of a photo of the device screen, e.g. img_px200_01.
    """
    return await asyncio.to_thread(repondre, question, photo_ref)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
