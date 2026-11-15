import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import requests

from core.AgentFlow import AgentFlow
from tools.ToolBuilder import build_tools


def execute(tools, name, **arguments):
    result = json.loads(tools.execute_tool(name, arguments))
    assert result.keys() == {"success", "error", "result"}
    assert result["success"], result["error"]
    assert result["error"] is None
    return result["result"]


def test_file_tools(tmp_path):
    tools = build_tools(math=False, web=False)
    source = str(tmp_path / "notatka.txt")
    destination = str(tmp_path / "przeniesiona.txt")
    directory = str(tmp_path)

    assert execute(tools, "create_file", path=source, content="pierwsza treść")["path"] == source
    assert execute(tools, "read_file", path=source)["content"] == "pierwsza treść"
    execute(tools, "write_file", path=source, content="druga treść")
    assert execute(tools, "search_in_files", path=directory, text="druga")["matches"][0]["line"] == "druga treść"
    assert execute(tools, "list_directory", path=directory)["entries"][0]["name"] == "notatka.txt"
    assert execute(tools, "get_file_info", path=source)["is_file"]
    assert execute(tools, "find_files", path=directory)["files"][0]["name"] == "notatka.txt"
    execute(tools, "move_file", source=source, destination=destination)
    assert Path(destination).read_text(encoding="utf-8") == "druga treść"
    execute(tools, "delete_file", path=destination)
    assert not Path(destination).exists()

    failure = json.loads(tools.execute_tool("read_file", {"path": destination}))
    assert failure["success"] is False
    assert "not found" in failure["error"]
    assert failure["result"] is None


@pytest.mark.parametrize(("name", "arguments", "field", "expected"), [
    ("calculate_expression", {"expression": "2 + 3"}, "result", "5"),
    ("solve_equation", {"equation": "x = 2"}, "solutions", ["2"]),
    ("differentiate_expression", {"expression": "x**2"}, "result", "2*x"),
    ("integrate_expression", {"expression": "2*x"}, "result", "x**2"),
    ("calculate_limit", {"expression": "sin(x)/x"}, "result", "1"),
])
def test_math_tools(name, arguments, field, expected):
    tools = build_tools(files=False, web=False)
    assert execute(tools, name, **arguments)[field] == expected


def test_web_tools():
    tools = build_tools(files=False, math=False)
    url = "https://example.com"
    with patch("tools.WebSearchTools.DDGS") as search:
        search.return_value.text.return_value = [{"title": "Przykład", "href": url}]
        assert url in execute(tools, "search_web", query="przykład")

    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Type": "text/html; charset=utf-8"}
    response.encoding = "utf-8"
    response.iter_content.return_value = [b"<article><p>Przykladowa tresc</p></article>"]
    with patch("tools.WebSearchTools.requests.get", return_value=response):
        assert "Przykladowa tresc" in execute(tools, "fetch_webpage_content", url=url)
    with patch("tools.WebSearchTools.requests.get", side_effect=requests.exceptions.Timeout):
        failure = json.loads(tools.execute_tool("fetch_webpage_content", {"url": url}))
    assert failure["success"] is False
    assert "timeout" in failure["error"]
    assert failure["result"] is None


def test_builder_and_shared_registry():
    assert build_tools(files=False, math=False, web=False).get_tools_list() == []
    assert len(build_tools().get_tools_list()) == 16
    tools = build_tools(files=False, web=False)
    flow = AgentFlow(tool_handle=tools)
    assert flow.ollama.ToolsHandle is flow.tools is tools
    for name, arguments in [
        ("read_file", {"path": "notatka.txt"}),
        ("calculate_expression", {}),
        ("calculate_expression", {"expression": "invalid()"}),
    ]:
        failure = json.loads(tools.execute_tool(name, arguments))
        assert failure.keys() == {"success", "error", "result"}
        assert failure["success"] is False
        assert failure["error"]
        assert failure["result"] is None


@pytest.mark.skipif(not os.environ.get("OLLAMA_TEST_MODEL"), reason="Ustaw OLLAMA_TEST_MODEL.")
@pytest.mark.parametrize("stream", [False, True])
def test_ollama_reads_notatka(tmp_path, monkeypatch, stream):
    monkeypatch.chdir(tmp_path)
    marker = f"NLP-{uuid4().hex}"
    Path("notatka.txt").write_text(marker, encoding="utf-8")
    tools = build_tools(math=False, web=False)
    for schema in tools.get_tools_list():
        if schema["function"]["name"] != "read_file":
            tools.delete_tool(schema["function"]["name"])

    flow = AgentFlow(
        base_url=os.environ.get("OLLAMA_TEST_ENDPOINT", "http://localhost:11434/v1"),
        tool_handle=tools,
        max_iterations=4,
        system_prompt="Czytaj wskazany plik narzędziem i przytocz jego dokładną treść.",
    )
    flow.ollama.client = flow.ollama.client.with_options(timeout=90, max_retries=0)
    assert flow.ollama.SetModel(os.environ["OLLAMA_TEST_MODEL"])

    with patch.object(tools, "execute_tool", wraps=tools.execute_tool) as executed:
        prompt = "odczytaj notatka.txt"
        if stream:
            events = flow.RunStream(prompt, as_events=True)
            answer = "".join(event["text"] for event in events if event["type"] == "token")
        else:
            answer = flow.Run(prompt)

    assert any(
        name == "read_file" and Path(arguments["path"]).resolve() == tmp_path / "notatka.txt"
        for name, arguments in (call.args for call in executed.call_args_list)
    )
    outputs = [json.loads(msg["content"]) for msg in flow.context if msg["role"] == "tool"]
    assert any(result["success"] and result["result"]["content"] == marker for result in outputs)
    assert marker in answer
