import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import Attachments, WebGui
from core.Attachments import AttachmentStore
from core.ChatStore import ChatStore, SettingsStore


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeAgent:
    def __init__(self):
        self.calls = []

    def load_history(self, messages):
        pass

    def stream(self, user_text, attachments=None):
        self.calls.append((user_text, attachments))
        yield {"type": "token", "text": "ok"}


class LegacyAgent(FakeAgent):
    # agent bez parametru attachments - wiadomość tekstowa nadal musi działać
    def stream(self, user_text):
        self.calls.append((user_text, None))
        yield {"type": "token", "text": "ok"}


class MermaidRepairAgent(FakeAgent):
    def repair_mermaid(self, source, error):
        self.repair_request = (source, error)
        return "```mermaid\nsequenceDiagram\n    Note over User,Server: Logowanie\n```"


@pytest.fixture
def gui(tmp_path, monkeypatch):
    agents = []

    def factory(settings, agent_class=FakeAgent):
        agent = agent_class()
        agents.append(agent)
        return agent

    monkeypatch.setattr(WebGui, "store", ChatStore(tmp_path / "chats"))
    monkeypatch.setattr(WebGui, "settings_store", SettingsStore(tmp_path / "settings.json"))
    monkeypatch.setattr(WebGui, "attachment_store", AttachmentStore(tmp_path / "attachments"))
    monkeypatch.setattr(WebGui, "agent_factory", factory)
    WebGui.sessions.clear()
    client = TestClient(WebGui.app)
    client.agents = agents
    client.tmp_path = tmp_path
    yield client
    WebGui.sessions.clear()


def upload(client, data=None, name="kot.png", mime="image/png"):
    return client.post(
        "/api/attachments", files={"file": (name, data if data is not None else png_bytes(), mime)}
    )


def send(client, **body):
    response = client.post("/api/chat", json=body)
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    return response, events


def test_upload_returns_public_record_without_paths(gui):
    response = upload(gui)
    assert response.status_code == 200
    record = response.json()
    assert record.keys() == {"id", "mime", "size", "width", "height", "name"}
    assert (record["mime"], record["width"], record["height"], record["name"]) == (
        "image/png", 4, 3, "kot.png",
    )


def test_invalid_mermaid_is_repaired_and_saved(gui):
    chat = WebGui.store.create()
    original = 'sequenceDiagram\n    note "Logowanie"'
    WebGui.store.append(chat["id"], "user", "Narysuj diagram logowania")
    WebGui.store.append(chat["id"], "assistant", f"```mermaid\n{original}\n```")
    agent = MermaidRepairAgent()
    WebGui.sessions[chat["id"]] = agent

    response = gui.post(
        f"/api/chats/{chat['id']}/mermaid/repair",
        json={"source": original, "error": "Expecting 'over'"},
    )
    assert response.status_code == 200
    corrected = response.json()["source"]
    assert "Note over User,Server: Logowanie" in corrected
    assert agent.repair_request == (original, "Expecting 'over'")

    saved = gui.put(
        f"/api/chats/{chat['id']}/mermaid",
        json={"original_source": original, "corrected_source": corrected},
    )
    assert saved.status_code == 200
    assert corrected in WebGui.store.get(chat["id"])["messages"][-1]["content"]


@pytest.mark.parametrize("data, status, code", [
    (b"<svg xmlns='http://www.w3.org/2000/svg'/>", 415, "bad_format"),
    (b"", 400, "empty"),
])
def test_upload_rejects_invalid_files(gui, data, status, code):
    response = upload(gui, data=data, name="x.png")
    assert response.status_code == status
    assert response.json()["code"] == code


def test_upload_rejects_too_large(gui, monkeypatch):
    monkeypatch.setattr(Attachments, "MAX_BYTES", 10)
    response = upload(gui)
    assert response.status_code == 413
    assert response.json()["code"] == "too_large"


def test_chat_binds_attachments_and_serves_them(gui):
    data = png_bytes()
    attachment_id = upload(gui, data=data).json()["id"]

    response, events = send(gui, message="co to?", attachments=[attachment_id])
    assert response.status_code == 200
    chat_id = events[0]["id"]
    assert events[-1] == {"type": "done"}

    text, attachments = gui.agents[0].calls[0]
    assert text == "co to?"
    assert [a.id for a in attachments] == [attachment_id]
    assert attachments[0].chat_id == chat_id

    user_message = gui.get(f"/api/chats/{chat_id}").json()["messages"][0]
    assert user_message["attachments"] == [attachment_id]

    served = gui.get(f"/api/chats/{chat_id}/attachments/{attachment_id}")
    assert served.status_code == 200
    assert served.content == data
    assert served.headers["content-type"] == "image/png"
    assert served.headers["x-content-type-options"] == "nosniff"

    # przypisany załącznik nie jest już oczekujący
    assert gui.delete(f"/api/attachments/{attachment_id}").status_code == 404


def test_image_only_message_is_allowed_but_empty_message_is_not(gui):
    attachment_id = upload(gui).json()["id"]
    response, _ = send(gui, message="  ", attachments=[attachment_id])
    assert response.status_code == 200

    response, _ = send(gui, message="  ")
    assert response.status_code == 400


def test_unknown_attachment_creates_no_chat(gui):
    response = gui.post("/api/chat", json={"message": "hej", "attachments": ["c" * 32]})
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert gui.get("/api/chats").json() == []


def test_partial_failure_binds_nothing(gui):
    good = upload(gui).json()["id"]
    response = gui.post("/api/chat", json={"message": "hej", "attachments": [good, "c" * 32]})
    assert response.status_code == 404
    # poprawny załącznik dalej czeka i można go usunąć jako oczekujący
    assert gui.delete(f"/api/attachments/{good}").json() == {"ok": True}


def test_too_many_attachments(gui, monkeypatch):
    monkeypatch.setattr(Attachments, "MAX_PER_MESSAGE", 1)
    ids = [upload(gui).json()["id"] for _ in range(2)]
    response = gui.post("/api/chat", json={"message": "hej", "attachments": ids})
    assert response.status_code == 400
    assert response.json()["code"] == "too_many"


def test_attachment_is_not_served_from_other_chat(gui):
    attachment_id = upload(gui).json()["id"]
    _, events = send(gui, message="a", attachments=[attachment_id])
    _, other = send(gui, message="b")
    response = gui.get(f"/api/chats/{other[0]['id']}/attachments/{attachment_id}")
    assert response.status_code == 404


def test_delete_chat_removes_its_attachments(gui):
    attachment_id = upload(gui).json()["id"]
    _, events = send(gui, message="a", attachments=[attachment_id])
    chat_id = events[0]["id"]
    assert (gui.tmp_path / "attachments" / chat_id).is_dir()

    assert gui.delete(f"/api/chats/{chat_id}").json() == {"ok": True}
    assert not (gui.tmp_path / "attachments" / chat_id).exists()


def test_text_message_works_with_agent_without_attachments_param(gui, monkeypatch):
    agents = []
    monkeypatch.setattr(WebGui, "agent_factory", lambda settings: agents.append(LegacyAgent()) or agents[-1])
    response, events = send(gui, message="hej")
    assert response.status_code == 200
    assert {"type": "token", "text": "ok"} in events
    assert agents[0].calls == [("hej", None)]


def test_settings_expose_attachment_limits(gui):
    limits = gui.get("/api/settings").json()["attachment_limits"]
    assert limits["max_bytes"] == Attachments.MAX_BYTES
    assert limits["max_per_message"] == Attachments.MAX_PER_MESSAGE
    assert "image/png" in limits["mime_types"]
    assert "image/svg+xml" not in limits["mime_types"]
