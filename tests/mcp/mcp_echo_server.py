"""Minimalny serwer MCP używany w testach integracyjnych.

Uruchamiany jako osobny proces:

    python tests/mcp/mcp_echo_server.py

"""

from __future__ import annotations

import os
import time

from mcp.server.mcpserver import MCPServer


server = MCPServer("echo-test")


@server.tool()
def echo(text: str) -> str:
    """Zwraca przekazany tekst bez zmian."""
    return text


@server.tool()
def add(a: int, b: int) -> int:
    """Dodaje dwie liczby całkowite."""
    return a + b


@server.tool()
def slow(seconds: float = 1.0) -> str:
    """Czeka podaną liczbę sekund, żeby przetestować timeouty."""
    time.sleep(seconds)
    return "done"


@server.tool()
def fail(message: str = "boom") -> str:
    """Zgłasza kontrolowany błąd narzędzia."""
    raise ValueError(message)


@server.tool()
def crash() -> str:
    """Kończy proces serwera, żeby przetestować reconnect."""
    os._exit(1)


if __name__ == "__main__":
    server.run(transport="stdio")
