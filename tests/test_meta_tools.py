"""Testy meta-narzędzia list_tools (tool prompt vs. katalog narzędzi).

Kontekst narzędziowy agenta składa się teraz z trzech elementów:
1. schemy ``tools`` (nazwa + krótki opis + typy parametrów) wysyłanej w requestie,
2. kilku linii zasad w promptcie systemowym (core/PromptBuilder.py),
3. narzędzia ``list_tools``, które na żądanie zwraca pełne opisy i parametry.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handlers.OllamaHandle import OllamaHandler
from handlers.ToolHandle import ToolHandle, short_tool_description
from tools.MetaTools import LIST_TOOLS_NAME, register_meta_tools
from tools.ToolBuilder import build_tools


def _registry() -> ToolHandle:
    """Rejestr jak w ToolBuilder: najpierw list_tools, potem narzędzia."""
    tools = ToolHandle()
    register_meta_tools(tools)
    tools.create_tool(
        "read_note",
        "Reads a stored note by id.",
        properties={"id": {"type": "string", "description": "Identifier of the note"}},
        required=["id"],
        function=lambda id: {"success": True, "note": f"note {id}"},
    )
    return tools


def test_list_tools_is_registered_first_and_visible_in_schema() -> None:
    tools = _registry()

    assert tools.tool_names()[0] == LIST_TOOLS_NAME
    schema = tools.get_tool(LIST_TOOLS_NAME)["function"]
    properties = schema["parameters"]["properties"]
    assert "tool_name" in properties
    assert schema["parameters"]["required"] == []


def test_list_tools_without_arguments_returns_names_only() -> None:
    """Przegląd to nazwy - krótkie opisy model widzi w listie ``tools``."""
    tools = _registry()
    raw = tools.execute_tool(LIST_TOOLS_NAME, {})
    payload = json.loads(raw)

    assert payload["success"] is True
    result = payload["result"]
    assert result["count"] == 2
    assert sorted(result["tools"]) == [LIST_TOOLS_NAME, "read_note"]
    # Brak dublowania: opisy narzędzi nie wracają w przeglądzie.
    assert "Reads a stored note" not in raw
    assert "tool_name=" in result["hint"]


def test_list_tools_with_tool_name_returns_full_parameter_details() -> None:
    tools = _registry()
    payload = json.loads(
        tools.execute_tool(LIST_TOOLS_NAME, {"tool_name": "read_note"})
    )

    record = payload["result"]["tool"]
    assert record["name"] == "read_note"
    assert record["description"] == "Reads a stored note by id."
    assert record["required"] == ["id"]
    # Opisy parametrów żyją tylko w rejestrze: kompaktowa lista wysyłana
    # modelowi je pomija, więc list_tools jest jedynym źródłem tej wiedzy.
    assert record["parameters"]["id"]["description"] == "Identifier of the note"
    assert record["parameters"]["id"]["type"] == "string"


def test_list_tools_unknown_tool_lists_the_known_ones() -> None:
    tools = _registry()
    payload = json.loads(tools.execute_tool(LIST_TOOLS_NAME, {"tool_name": "nope"}))

    assert payload["success"] is False
    assert payload["result"] is None
    error = payload["error"]
    assert "nie jest zarejestrowane" in error
    assert "read_note" in error and LIST_TOOLS_NAME in error


def test_list_tools_follows_the_unified_description_helper() -> None:
    """Rejestr, schema tools i UI skracają opisy jednym helperem."""
    tools = build_tools(workspace_dir=".")
    handler = OllamaHandler(tool_handle=tools)
    compact = {
        item["function"]["name"]: item["function"]["description"]
        for item in handler._CompactTools(tools.get_tools_list())
    }

    assert set(compact) == set(tools.tool_names())
    for name, description in compact.items():
        assert description == short_tool_description(
            tools.get_tool(name)["function"]["description"]
        )


def test_introspection_is_free_even_when_reads_require_consent() -> None:
    tools = ToolHandle(risk_policies={"read": "ask"})
    register_meta_tools(tools)

    prepared = tools.prepare_tool_call(LIST_TOOLS_NAME, {})
    assert prepared.policy == "allow"
    assert prepared.requires_approval is False
    assert json.loads(tools.execute_tool(LIST_TOOLS_NAME, {}))["success"] is True

    # Jawna polityka z settings.json nadal ma pierwszeństwo nad spec toola.
    tools.set_policy_config(policy_overrides={LIST_TOOLS_NAME: "deny"})
    denied = json.loads(tools.execute_tool(LIST_TOOLS_NAME, {}))
    assert denied["success"] is False
    assert denied["result"]["denied"] is True


def test_introspection_results_fit_the_tool_result_budget() -> None:
    """AgentFlow docina rezultaty do DEFAULT_TOOL_CHARS.

    Gdyby list_tools nie mieścił się w tym limitie, TruncateJson urwałby JSON
    w połowie i model zobaczyłby nieparsowalny tekst z [TRUNCATED].
    """
    from core.AgentFlow import DEFAULT_TOOL_CHARS

    tools = build_tools(workspace_dir=".")
    calls = [(LIST_TOOLS_NAME, {})] + [
        (LIST_TOOLS_NAME, {"tool_name": name}) for name in tools.tool_names()
    ]

    for name, arguments in calls:
        raw = tools.execute_tool(name, arguments)
        assert len(raw) <= DEFAULT_TOOL_CHARS, f"{arguments}: {len(raw)} znaków"
        assert "[TRUNCATED]" not in raw
        json.loads(raw)  # poprawny JSON, nie urwany


def test_new_mcp_tools_become_visible_to_list_tools() -> None:
    """Rejestr jest dynamiczny - list_tools widzi narzędzia dołączone po startie."""
    tools = ToolHandle()
    register_meta_tools(tools)
    tools.create_tool_from_schema(
        "server__weather",
        "MCP weather lookup.",
        {"type": "object", "properties": {}, "required": []},
        function=lambda: {"success": True, "temp": 3},
        source="mcp",
        server="server",
        remote_name="weather",
    )

    payload = json.loads(tools.execute_tool(LIST_TOOLS_NAME, {}))
    names = set(payload["result"]["tools"])
    assert "server__weather" in names
    record = json.loads(
        tools.execute_tool(LIST_TOOLS_NAME, {"tool_name": "server__weather"})
    )["result"]["tool"]
    assert record["source"] == "mcp"
    assert record["risk"] == [] or "external" in record["risk"]
