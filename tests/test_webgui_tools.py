import json
import time
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from core.WebGui import create_app
from core.main import build_api_runtime
from core.RuntimeConfig import RuntimeConfig


class FakeProvider:
    def __init__(self, config, tools):
        self.config = config
        self.ToolsHandle = tools
        self.base_url = config.endpoint
        self.model = "test-model"
        self.closed = 0

    def GetModels(self):
        return [NS(id="test-model")]

    def SetModel(self, name):
        self.model = name
        return True

    def GetNumCtx(self):
        return 4096

    def EnsureContextBudget(self, messages, **kwargs):
        return None

    def ChatStream(self, messages, **kwargs):
        yield NS(choices=[NS(delta=NS(content="ok", tool_calls=None, reasoning_content=None))])

    def close(self):
        self.closed += 1


def stream_delta(content=None, tool_calls=None):
    return NS(choices=[NS(delta=NS(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    ))])


def stream_fragment(name, arguments):
    return NS(
        id="call-1",
        index=0,
        function=NS(name=name, arguments=json.dumps(arguments)),
    )


class ScriptedProvider(FakeProvider):
    def __init__(self, config, tools, scripts):
        super().__init__(config, tools)
        self.scripts = list(scripts)

    def ChatStream(self, messages, **kwargs):
        yield from self.scripts.pop(0)


class ToolAgent:
    def __init__(self):
        self.calls = []

    def load_history(self, messages):
        pass

    def stream(self, text, attachments=None):
        self.calls.append(text)
        yield {
            "type": "tool_start",
            "tool": "read_file",
            "source": "builtin",
            "server": None,
            "risk": ["read"],
        }
        yield {
            "type": "tool_result",
            "tool": "read_file",
            "source": "builtin",
            "server": None,
            "risk": ["read"],
            "status": "ok",
        }
        yield {"type": "token", "text": "odpowiedź"}


class ResetAnswerAgent(ToolAgent):
    def stream(self, text, attachments=None):
        self.calls.append(text)
        yield {"type": "token", "text": "zła odpowiedź"}
        yield {"type": "answer_reset", "reason": "tool_json"}
        yield {"type": "token", "text": "dobra odpowiedź"}


class DeniedToolAgent(ToolAgent):
    def stream(self, text, attachments=None):
        self.calls.append(text)
        yield {
            "type": "tool_result",
            "tool": "search_web",
            "source": "builtin",
            "server": None,
            "risk": ["read", "external"],
            "status": "denied",
            "code": "invalid_arguments",
        }
        yield {"type": "token", "text": "nie udało się"}


def setup(tmp_path):
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        model="test-model",
        api_key="private-secret",
    )
    runtime = build_api_runtime(config, provider_factory=FakeProvider)
    app = create_app(runtime=runtime)
    return app, runtime


def test_tools_endpoint_exposes_risk_and_policy(tmp_path):
    app, runtime = setup(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/tools")
            assert response.status_code == 200
            data = response.json()
            assert str(tmp_path) in " ".join(data["allowed_directories"])
            tools = {item["name"]: item for item in data["tools"]}
            assert tools["read_file"]["risk"] == ["read"]
            assert tools["read_file"]["policy"] == "allow"
            assert "destructive" in tools["delete_file"]["risk"]
            assert tools["delete_file"]["policy"] == "ask"
            assert tools["ask_user"]["kind"] == "user_question"
    finally:
        runtime.close()


def test_policy_endpoint_validates_and_applies(tmp_path):
    app, runtime = setup(tmp_path)
    try:
        with TestClient(app) as client:
            invalid = client.put("/api/tools/policies", json={
                "policies": {"no_such_tool": "deny"},
            })
            assert invalid.status_code == 400

            updated = client.put("/api/tools/policies", json={
                "policies": {"delete_file": "allow"},
                "risk_policies": {"external": "deny"},
            })
            assert updated.status_code == 200
            assert runtime.tool_handle.get_policy("delete_file") == "allow"
            assert runtime.tool_handle.get_policy("search_web") == "deny"

            tools = {item["name"]: item for item in client.get("/api/tools").json()["tools"]}
            assert tools["delete_file"]["policy"] == "allow"
            assert tools["search_web"]["policy"] == "deny"

            assert client.put("/api/settings", json={"tool_policies": {"read_file": "deny"}}).status_code == 400
    finally:
        runtime.close()


def test_approval_endpoint_status_codes(tmp_path):
    app, runtime = setup(tmp_path)
    try:
        with TestClient(app) as client:
            broker = runtime.interaction_broker
            pending = broker.register_approval(
                tool="delete_file", risk=("write", "destructive"),
                arguments={"path": "a.txt"}, chat_id="chat", run_id="run",
            )
            assert client.post(f"/api/approvals/{pending.id}", json={"decision": "allow_once"}).status_code == 200
            assert client.post(f"/api/approvals/{pending.id}", json={"decision": "allow_once"}).status_code == 409
            assert client.post("/api/approvals/apr_missing", json={"decision": "deny"}).status_code == 404

            invalid = broker.register_approval(
                tool="x", risk=("write",), arguments={}, chat_id="chat", run_id="r2",
            )
            assert client.post(f"/api/approvals/{invalid.id}", json={"decision": "maybe"}).status_code == 400

            expired = broker.register_approval(
                tool="x", risk=("write",), arguments={}, chat_id="chat", run_id="r3",
                expires_in=0.01,
            )
            time.sleep(0.03)
            assert client.post(f"/api/approvals/{expired.id}", json={"decision": "deny"}).status_code == 410
    finally:
        runtime.close()


def test_question_endpoint_option_and_custom_answer(tmp_path):
    app, runtime = setup(tmp_path)
    try:
        with TestClient(app) as client:
            broker = runtime.interaction_broker
            first = broker.register_question(
                question="Wariant?", options=["A", "B"], chat_id="chat", run_id="run",
            )
            response = client.post(f"/api/questions/{first.id}", json={
                "answer": "B", "option_index": 1,
            })
            assert response.status_code == 200
            assert broker.poll(first) == {"answer": "B", "option": "B"}

            second = broker.register_question(
                question="Wariant?", options=["A", "B"], chat_id="chat", run_id="run2",
            )
            response = client.post(f"/api/questions/{second.id}", json={"answer": "własna"})
            assert response.status_code == 200
            assert broker.poll(second) == {"answer": "własna"}

            third = broker.register_question(
                question="Wariant?", options=["A", "B"], chat_id="chat", run_id="run3",
            )
            assert client.post(f"/api/questions/{third.id}", json={"cancelled": True}).status_code == 200
            assert broker.poll(third) == {"cancelled": True}
    finally:
        runtime.close()


def test_webgui_streams_search_web_approval_and_drops_extra_args(tmp_path):
    config = RuntimeConfig(
        data_dir=tmp_path / "data", workspace=tmp_path,
        model="test-model", api_key="private-secret",
    )
    scripts = [
        [stream_delta(tool_calls=[stream_fragment(
            "search_web", {"query": "kurs dolara", "region": "pl"}
        )])],
        [stream_delta(content="Nie mam zgody na wyszukiwanie.")],
    ]
    made = []

    def make(config, tools):
        provider = ScriptedProvider(config, tools, scripts)
        made.append(provider)
        return provider

    runtime = build_api_runtime(config, provider_factory=make)
    app = create_app(runtime=runtime)
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST", "/api/chat", json={"message": "aktualna cena dolara"}
            ) as response:
                assert response.status_code == 200
                events = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    events.append(event)
                    if event.get("type") == "approval_required":
                        assert event["tool"] == "search_web"
                        assert event["arguments"]["query"] == "kurs dolara"
                        assert "region" not in event["arguments"]
                        assert runtime.interaction_broker.resolve_approval(
                            event["id"], "deny"
                        ).ok
                    if event.get("type") == "done":
                        break
        assert any(e["type"] == "tool_result" and e["status"] == "denied" for e in events)
        assert any(e["type"] == "token" and "Nie mam zgody" in e["text"] for e in events)
    finally:
        runtime.close()


def test_chat_answer_reset_discards_draft(tmp_path):
    app, runtime = setup(tmp_path)
    runtime.agent_factory = lambda settings: ResetAnswerAgent()
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": "zrób"})
            assert response.status_code == 200
            events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            assert any(event["type"] == "answer_reset" for event in events)
            chat_id = next(event["id"] for event in events if event["type"] == "chat_id")
            stored = client.get(f"/api/chats/{chat_id}").json()["messages"][-1]
            assert stored["content"] == "dobra odpowiedź"
    finally:
        runtime.close()


def test_chat_persists_tool_error_code(tmp_path):
    app, runtime = setup(tmp_path)
    runtime.agent_factory = lambda settings: DeniedToolAgent()
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": "kurs dolara"})
            assert response.status_code == 200
            events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            chat_id = next(event["id"] for event in events if event["type"] == "chat_id")
            stored = client.get(f"/api/chats/{chat_id}").json()["messages"][-1]
            assert stored["tools"][0]["code"] == "invalid_arguments"
            assert stored["tools"][0]["status"] == "denied"
    finally:
        runtime.close()


def test_chat_persists_tool_badges_and_streams_events(tmp_path):
    app, runtime = setup(tmp_path)
    runtime.agent_factory = lambda settings: ToolAgent()
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": "hej"})
            assert response.status_code == 200
            events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            assert any(event["type"] == "tool_start" for event in events)
            assert any(event["type"] == "tool_result" for event in events)
            chat_id = next(event["id"] for event in events if event["type"] == "chat_id")
            stored = client.get(f"/api/chats/{chat_id}").json()["messages"][-1]
            assert stored["tools"] == [{
                "name": "read_file",
                "source": "builtin",
                "server": None,
                "risk": ["read"],
                "status": "ok",
            }]
            assert "arguments" not in json.dumps(stored)
    finally:
        runtime.close()
