"""Serveur MCP (transport streamable-http) exposant deux tools.

Lancement : uv run python -m c_mcp.server          → http://127.0.0.1:8000/mcp
Variables : MCP_HOST (défaut 127.0.0.1), MCP_PORT (défaut 8000).
"""
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("serveur-demo", host=os.getenv("MCP_HOST", "127.0.0.1"),
              port=int(os.getenv("MCP_PORT", "8000")))

EMPLOYES = {
    "mohamed": {"name": "Mohamed", "salary": 12000, "currency": "MAD", "job": "Developer"},
    "fatima": {"name": "Fatima", "salary": 15000, "currency": "MAD", "job": "Data Engineer"},
}


@mcp.tool()
def get_employee_info(name: str) -> dict:
    """Get information (name, salary, job) about an employee, by first name."""
    employe = EMPLOYES.get(name.strip().lower())
    if employe is None:
        raise ValueError(f"Unknown employee: {name!r}")
    return employe


@mcp.tool()
def get_weather(location: str) -> str:
    """Get the weather for a location (simulated data)."""
    return f"Weather in {location}: Sunny, 32°C (simulated)"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
