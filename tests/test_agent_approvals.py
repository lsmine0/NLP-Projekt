"""End-to-end approval/question flow tests using the real AgentFlow loop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

from core.AgentAdapter import OllamaAgent
from core.AuditLog import AuditLog
from core.Interactions import InteractionBroker
from handlers.ToolHandle import ToolHandle
from tools.CodeTools import register_code_tools
from tools.InteractionTools import register_interaction_tools


def delta(content=None, fragments=None, reasoning=None):
    return NS(choices=[NS(delta=NS(
        content=content,
        reasoning_content=reasoning,
        reasoning=reasoning,
        tool_calls=fragments,
    ))])


def fragment(call_id, name, arguments, index=0):
    return NS(
        id=call_id,
        index=index,
        function=NS(name=name, arguments=json.dumps(arguments)),
    )


def answer(text):
    return [delta(content=text)]


class ScriptProvider:
    def __init__(self, tools, scripts):
        self.ToolsHandle = tools
        self.base_url = "http://fake/v1"
        self.model = "scripted-model"
        self.scripts = list(scripts)
        self.payloads = []
        self.retry_answer = ""
        self.retry_calls = 0

    def GetModels(self):
        return [NS(id=self.model)]

    def SetModel(self, name):
        self.model = name
        return True

    def GetNumCtx(self):
        return 8192

    def EnsureContextBudget(self, messages, **kwargs):
        return None

    def supports_vision(self):
        return True

    def ChatStream(self, messages, **kwargs):
        self.payloads.append(json.loads(json.dumps(messages)))
        script = self.scripts.pop(0) if self.scripts else answer("OK")
        yield from script

    def ChatNoTools(self, messages, **kwargs):
        self.retry_calls += 1
        return NS(choices=[NS(message=NS(content=self.retry_answer))])

    def close(self):
        pass


def make_agent(tmp_path, tools, scripts, broker):
    provider = ScriptProvider(tools, scripts)
    agent = OllamaAgent(
        "http://fake/v1",
        "",
        system_prompt="SYS",
        tool_handle=tools,
        provider=provider,
        interaction_broker=broker,
    )
    return agent


def make_write_tool(tools, calls, *, risk=("write",), path_args=("path",)):
    def probe(path, token="x"):
        calls.append({"path": path, "token": token})
        return {"success": True, "path": path}

    tools.create_tool(
        "probe",
        "probe",
        properties={
            "path": {"type": "string"},
            "token": {"type": "string"},
        },
        required=["path"],
        function=probe,
        risk=risk,
        path_args=path_args,
    )


def run_with_decisions(agent, broker, prompt="zrób", decisions=None):
    decisions = list(decisions or [])
    events = []
    for event in agent.stream(prompt):
        events.append(event)
        if event["type"] == "approval_required":
            decision = decisions.pop(0) if decisions else "deny"
            assert broker.resolve_approval(event["id"], decision).ok
    return events


def test_allow_once_executes_and_continues(tmp_path):
    calls = []
    tools = ToolHandle(allowed_directories=[tmp_path], primary_root=tmp_path)
    make_write_tool(tools, calls)
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {
            "path": "a.txt",
            "token": "super-secret-token",
        })])],
        answer("gotowe"),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)

    events = []
    for event in agent.stream("zrób"):
        events.append(event)
        if event["type"] == "approval_required":
            assert event["arguments"]["token"] == "***"
            assert broker.resolve_approval(event["id"], "allow_once").ok

    assert [event["type"] for event in events].count("approval_required") == 1
    assert any(event["type"] == "tool_result" and event["status"] == "ok" for event in events)
    assert calls and Path(calls[0]["path"]) == (tmp_path / "a.txt").resolve()
    assert "gotowe" in "".join(e["text"] for e in events if e["type"] == "token")


def test_run_python_executes_after_allow_once(tmp_path):
    """Regression test: po zezwoleniu użytkownika run_python musi się naprawdę uruchomić.

    Host nie konfiguruje approval_callback ani allow_execution - zgoda płynie tylko
    z polityki rejestru (ask -> approval_required -> allow_once).
    """
    tools = ToolHandle()
    register_code_tools(tools, workspace_dir=tmp_path)
    scripts = [
        [delta(fragments=[fragment("c1", "run_python", {"code": "print('po zgody')"})])],
        answer("Kod wykonany."),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)

    events = []
    for event in agent.stream("uruchom kod"):
        events.append(event)
        if event["type"] == "approval_required":
            assert event["tool"] == "run_python"
            assert broker.resolve_approval(event["id"], "allow_once").ok

    tool_messages = [m for m in agent.flow.context if m.get("role") == "tool"]
    payload = json.loads(tool_messages[-1]["content"])
    assert payload["success"] is True, payload.get("error")
    assert payload["result"]["stdout"].strip() == "po zgody"
    assert any(
        event["type"] == "tool_result" and event["status"] == "ok"
        for event in events
    )


def test_denial_is_structured_and_agent_can_continue(tmp_path):
    calls = []
    tools = ToolHandle(allowed_directories=[tmp_path], primary_root=tmp_path)
    make_write_tool(tools, calls)
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {"path": "a.txt"})])],
        answer("nie mogę"),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    agent.flow.audit_log = audit

    events = run_with_decisions(agent, broker, decisions=["deny"])

    assert not calls
    denied = [e for e in events if e["type"] == "tool_result" and e["status"] == "denied"]
    assert denied and denied[0]["code"] == "user_denied"
    tool_msg = [m for m in agent.flow.context if m.get("role") == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["success"] is False
    assert payload["result"]["denied"] is True
    assert "nie mogę" in "".join(e["text"] for e in events if e["type"] == "token")
    audit_lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert any(
        line["kind"] == "decision"
        and line["tool"] == "probe"
        and line["decision"] == "deny"
        and line["reason"] == "user"
        for line in audit_lines
    )
    assert "a.txt" not in audit_path.read_text(encoding="utf-8")


def test_session_allow_skips_second_prompt(tmp_path):
    calls = []
    tools = ToolHandle(allowed_directories=[tmp_path], primary_root=tmp_path)
    make_write_tool(tools, calls)
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {"path": "a.txt"})])],
        answer("pierwsze"),
        [delta(fragments=[fragment("c2", "probe", {"path": "b.txt"})])],
        answer("drugie"),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)

    first = run_with_decisions(agent, broker, decisions=["allow_session"])
    second = run_with_decisions(agent, broker, decisions=[])

    assert any(e["type"] == "approval_required" for e in first)
    assert not any(e["type"] == "approval_required" for e in second)
    assert [Path(call["path"]).name for call in calls] == ["a.txt", "b.txt"]


def test_approval_timeout_returns_denial(tmp_path):
    tools = ToolHandle(allowed_directories=[tmp_path], primary_root=tmp_path)
    make_write_tool(tools, [])
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {"path": "a.txt"})])],
        answer("po timeoucie"),
    ]
    broker = InteractionBroker(timeout=0.05)
    agent = make_agent(tmp_path, tools, scripts, broker)

    events = list(agent.stream("zrób"))
    denied = [e for e in events if e["type"] == "tool_result" and e["status"] == "denied"]
    assert denied and denied[0]["code"] == "approval_timeout"
    assert any(e["type"] == "token" and "timeoucie" in e["text"] for e in events)


def test_client_disconnect_cancels_pending_approval(tmp_path):
    tools = ToolHandle(allowed_directories=[tmp_path], primary_root=tmp_path)
    make_write_tool(tools, [])
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {"path": "a.txt"})])],
        answer("po rozłączeniu"),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)

    events = []
    for event in agent.stream("zrób"):
        events.append(event)
        if event["type"] == "approval_required":
            broker.cancel_run(agent.flow.run_id)

    denied = [e for e in events if e["type"] == "tool_result" and e["status"] == "denied"]
    assert denied and denied[0]["code"] == "client_disconnected"


def test_ask_user_option_and_custom_answer(tmp_path):
    tools = ToolHandle()
    register_interaction_tools(tools)
    scripts = [
        [delta(fragments=[fragment("q1", "ask_user", {
            "question": "Który wariant?",
            "options": ["A", "B"],
        })])],
        answer("wybrano"),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)

    events = []
    for event in agent.stream("zapytaj"):
        events.append(event)
        if event["type"] == "user_input_required":
            assert event["options"] == ["A", "B"]
            assert broker.resolve_question(event["id"], {"answer": "B", "option": "B"}).ok

    tool_msg = [m for m in agent.flow.context if m.get("role") == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["success"] is True
    assert payload["result"] == {"answer": "B", "option": "B"}
    assert any(e["type"] == "token" and "wybrano" in e["text"] for e in events)


def test_audit_log_has_decisions_and_destructive_without_secrets(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    calls = []
    tools = ToolHandle(
        allowed_directories=[tmp_path],
        primary_root=tmp_path,
        policy_overrides={"probe": "allow"},
    )
    make_write_tool(tools, calls, risk=("write", "destructive"))
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {
            "path": "a.txt",
            "token": "super-secret-token",
        })])],
        answer("ok"),
    ]
    broker = InteractionBroker(timeout=5)
    agent = make_agent(tmp_path, tools, scripts, broker)
    agent.flow.audit_log = audit

    list(agent.stream("zrób"))
    lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert any(line["kind"] == "destructive" and line["tool"] == "probe" for line in lines)
    assert any(line.get("success") is True for line in lines if line["kind"] == "destructive")
    assert "super-secret-token" not in audit_path.read_text(encoding="utf-8")
    assert all("path" not in line and "arguments" not in line for line in lines)


def test_empty_answer_after_tool_is_retried(tmp_path):
    tools = ToolHandle()
    tools.create_tool("probe", "probe", function=lambda: {"success": True})
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {})])],
        [delta(reasoning="Tylko myślenie, bez treści.")],
    ]
    broker = InteractionBroker(timeout=5)
    provider = ScriptProvider(tools, scripts)
    provider.retry_answer = "Odpowiedź po ponowieniu."
    agent = OllamaAgent(
        "http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
        provider=provider, interaction_broker=broker,
    )

    events = list(agent.stream("zrób"))
    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert tokens == ["Odpowiedź po ponowieniu."]
    assert provider.retry_calls == 1
    assert agent.flow.context[-1]["role"] == "assistant"
    assert agent.flow.context[-1]["content"] == "Odpowiedź po ponowieniu."
    assert not any(
        message.get("role") == "assistant"
        and not message.get("content", "").strip()
        and not message.get("tool_calls")
        for message in agent.flow.context
    )


def test_empty_answer_after_tool_falls_back_to_error(tmp_path):
    tools = ToolHandle()
    tools.create_tool("probe", "probe", function=lambda: {"success": True})
    scripts = [
        [delta(fragments=[fragment("c1", "probe", {})])],
        [delta(reasoning="Tylko myślenie.")],
    ]
    broker = InteractionBroker(timeout=5)
    provider = ScriptProvider(tools, scripts)
    provider.retry_answer = ""
    agent = OllamaAgent(
        "http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
        provider=provider, interaction_broker=broker,
    )

    events = list(agent.stream("zrób"))
    assert any(event["type"] == "error" for event in events)
    assert provider.retry_calls == 1
    assert not any(
        message.get("role") == "assistant"
        and not message.get("content", "").strip()
        and not message.get("tool_calls")
        for message in agent.flow.context
    )


def test_load_history_skips_empty_assistant_after_think_strip(tmp_path):
    tools = ToolHandle()
    provider = ScriptProvider(tools, [])
    agent = OllamaAgent(
        "http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
        provider=provider,
    )
    agent.load_history([
        {"role": "assistant", "content": "<think>ukryte myślenie</think>"},
        {"role": "user", "content": "Cześć"},
        {"role": "assistant", "content": "Odpowiedź"},
    ])
    assert [message["role"] for message in agent.flow.context] == ["system", "user", "assistant"]
    assert agent.flow.context[-1]["content"] == "Odpowiedź"


def test_textual_tool_json_is_retried_with_tool_calls(tmp_path):
    calls = []
    tools = ToolHandle()
    tools.create_tool("probe", "probe", function=lambda: calls.append("ok") or {"success": True})
    bad_json = (
        "Zastosuję narzędzie probe.\n\n"
        "```json\n{\"tool\": \"probe\", \"arguments\": {}}\n```"
    )
    scripts = [
        [delta(content=bad_json)],
        [delta(fragments=[fragment("c1", "probe", {})])],
        answer("Gotowe."),
    ]
    broker = InteractionBroker(timeout=5)
    provider = ScriptProvider(tools, scripts)
    agent = OllamaAgent(
        "http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
        provider=provider, interaction_broker=broker,
    )

    events = list(agent.stream("zrób"))
    types = [event["type"] for event in events]
    assert "answer_reset" in types
    assert "warning" in types
    assert calls == ["ok"]
    assert any(event["type"] == "token" and event["text"] == "Gotowe." for event in events)
    context_text = json.dumps(agent.flow.context, ensure_ascii=False)
    assert "```json" not in context_text
    assert "Zastosuję narzędzie probe" not in context_text
    # Drugie zapytanie do modelu zawiera korektę, ale nie trafia ona do kontekstu.
    assert len(provider.payloads) >= 2
    correction = provider.payloads[1]
    assert any(
        message.get("role") == "system" and "tool_calls" in message.get("content", "")
        for message in correction
    )


def test_textual_tool_json_twice_returns_error(tmp_path):
    tools = ToolHandle()
    tools.create_tool("probe", "probe", function=lambda: {"success": True})
    bad_json = '```json\n{"tool": "probe", "arguments": {}}\n```'
    scripts = [
        [delta(content=bad_json)],
        [delta(content=bad_json)],
    ]
    broker = InteractionBroker(timeout=5)
    provider = ScriptProvider(tools, scripts)
    agent = OllamaAgent(
        "http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
        provider=provider, interaction_broker=broker,
    )

    events = list(agent.stream("zrób"))
    assert sum(1 for event in events if event["type"] == "answer_reset") == 2
    assert any(event["type"] == "error" for event in events)
    context_text = json.dumps(agent.flow.context, ensure_ascii=False)
    assert "```json" not in context_text
