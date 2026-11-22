"""Tests #51 (slices F+G): tools/ImageTools.py - inspekcja po ID, scope sesji, zero ścieżek."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.Attachments import AttachmentStore
from handlers.ToolHandle import ToolHandle, use_tool_session
from tools.ImageTools import list_attachments, register_tools, inspect_image

CHAT = "a" * 32
OTHER = "b" * 32


def png_bytes(size=(8, 6)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    return AttachmentStore(tmp_path / "attachments")


@pytest.fixture
def tools():
    handle = ToolHandle()
    register_tools(handle)
    return handle


def scope(store, chat_id, ids=()):
    return {"chat_id": chat_id, "attachment_store": store, "attachment_ids": list(ids)}


# --------------------------------------------------------------- inspect_image
def test_inspect_image_returns_metadata_without_paths(store):
    attachment = store.save(png_bytes((12, 9)), name="kot.png", chat_id=CHAT)
    with use_tool_session(scope(store, CHAT, [attachment.id])):
        result = inspect_image()
    assert result["success"] is True
    assert result["id"] == attachment.id
    assert (result["name"], result["mime_type"], result["width"], result["height"]) == (
        "kot.png", "image/png", 12, 9)
    assert result["aspect_ratio"] == round(12 / 9, 3)
    dumped = json.dumps(result)
    assert str(store.directory) not in dumped


def test_inspect_image_by_explicit_id_and_index(store):
    one = store.save(png_bytes((6, 4)), name="one.png", chat_id=CHAT)
    two = store.save(png_bytes((8, 6)), name="two.png", chat_id=CHAT)
    with use_tool_session(scope(store, CHAT, [two.id, one.id])):
        assert inspect_image(index=1)["id"] == two.id      # kolejność z bieżącej message
        assert inspect_image(attachment_id=one.id)["id"] == one.id


def test_inspect_image_out_of_range_and_bad_index(store):
    attachment = store.save(png_bytes(), chat_id=CHAT)
    with use_tool_session(scope(store, CHAT, [attachment.id])):
        out_of_range = inspect_image(index=5)
        assert out_of_range["success"] is False and "poza zakresem" in out_of_range["error"]
        bad = inspect_image(index="two")
        assert bad["success"] is False and "liczbą" in bad["error"]


def test_inspect_image_rejects_attachment_of_another_chat(store):
    foreign = store.save(png_bytes(), chat_id=OTHER)
    with use_tool_session(scope(store, CHAT)):
        result = inspect_image(attachment_id=foreign.id)
    assert result["success"] is False and "czacie" in result["error"]
    assert str(store.directory) not in json.dumps(result)


def test_inspect_image_without_session_scope(tools):
    with use_tool_session({}):
        result = inspect_image()
    assert result["success"] is False and "sesj" in result["error"]


# --------------------------------------------------------------- list_attachments
def test_list_attachments_is_scoped_to_the_chat(store):
    mine = store.save(png_bytes(), name="mine.png", chat_id=CHAT)
    theirs = store.save(png_bytes(), name="theirs.png", chat_id=OTHER)
    with use_tool_session(scope(store, CHAT, [mine.id])):
        result = list_attachments()
    assert result["success"] is True and result["count"] == 1
    assert [item["id"] for item in result["attachments"]] == [mine.id]
    assert theirs.id not in json.dumps(result)
    assert result["current_attachment_ids"] == [mine.id]


def test_list_attachments_empty_chat(store):
    with use_tool_session(scope(store, CHAT)):
        result = list_attachments()
    assert result["success"] is True and result["count"] == 0


# ------------------------------------------------- ToolHandle envelope + scope
def test_registration_and_envelope_shape(tools, store):
    names = {tool["function"]["name"] for tool in tools.get_tools_list()}
    assert names == {"inspect_image", "list_attachments"}
    with use_tool_session(scope(store, CHAT)):
        raw = tools.execute_tool("list_attachments", {})
    payload = json.loads(raw)
    assert payload["success"] is True and payload["result"]["count"] == 0
    assert payload["error"] is None


def test_session_scope_is_visible_inside_tool_thread(store):
    """AgentFlow executes tools w wątku ThreadPoolExecutor - scope musi być widoczny."""
    attachment = store.save(png_bytes(), name="kot.png", chat_id=CHAT)

    class ToolCallingProvider:
        def __init__(self, tools):
            self.ToolsHandle = tools
            self.base_url = "http://fake/v1"
            self.model = "vision-model"

        def GetModels(self):
            return [NS(id=self.model)]

        def SetModel(self, name):
            self.model = name
            return True

        def GetNumCtx(self):
            return 16384

        def EnsureContextBudget(self, messages, **kwargs):
            return None

        def supports_vision(self):
            return True

        def ChatStream(self, messages, **kwargs):
            ran_tools = any(message.get("role") == "tool" for message in messages)
            if not ran_tools:
                call = NS(id="call-1", function=NS(name="inspect_image", arguments='{"index": 1}'))
                yield NS(choices=[NS(delta=NS(content=None, tool_calls=[call],
                                            reasoning_content=None))])
                return
            yield NS(choices=[NS(delta=NS(content="Kot.", tool_calls=None,
                                        reasoning_content=None))])

        def close(self):
            pass

    from core.AgentAdapter import OllamaAgent

    tools = ToolHandle()
    register_tools(tools)
    provider = ToolCallingProvider(tools)
    agent = OllamaAgent("http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
                        provider=provider, attachment_store=store, chat_id=CHAT)
    list(agent.stream("Opisz", [attachment]))

    tool_messages = [message for message in agent.flow.context if message.get("role") == "tool"]
    assert tool_messages, "narzędzie nie zostało wywołane"
    payload = json.loads(tool_messages[0]["content"])
    assert payload["success"] is True
    assert payload["result"]["id"] == attachment.id
    assert str(store.directory) not in tool_messages[0]["content"]
