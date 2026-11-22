"""Tests #51 (slices A+B+C): multimodalny stream agenta, capability-gate i tokeny."""
import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.AgentAdapter import OllamaAgent
from core.Attachments import SEND_MAX_SIDE, AttachmentStore
from handlers.ContextHandle import ATTACHMENT_TOKENS, ContextHandle
from handlers.ToolHandle import ToolHandle

CHAT = "a" * 32


def png_bytes(size=(8, 6)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeProvider:
    """Atrapa providera: zapisuje payload i opcjonalnie zgłasza tool call."""

    def __init__(self, tools, vision=True):
        self.ToolsHandle = tools
        self.base_url = "http://fake/v1"
        self.model = "vision-model" if vision else "text-model"
        self.vision = vision
        self.payloads = []
        self.closed = 0

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
        return self.vision

    def ChatStream(self, messages, **kwargs):
        self.payloads.append(json.loads(json.dumps(messages)))
        ran_tools = any(message.get("role") == "tool" for message in messages)
        if ran_tools:
            yield NS(choices=[NS(delta=NS(content="Obraz opisany.", tool_calls=None,
                                        reasoning_content=None))])
            return
        yield NS(choices=[NS(delta=NS(content="Obraz opisany.", tool_calls=None,
                                    reasoning_content=None))])

    def close(self):
        self.closed += 1


def build_agent(vision=True, store=None, chat_id=CHAT):
    tools = ToolHandle()
    provider = FakeProvider(tools, vision=vision)
    agent = OllamaAgent("http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
                        provider=provider, attachment_store=store,
                        chat_id=(chat_id if store else None))
    return agent, provider


@pytest.fixture
def store(tmp_path):
    return AttachmentStore(tmp_path / "attachments")


# ---------------------------------------------------------- multimodal stream
def test_stream_sends_data_url_but_keeps_ids_in_context(store):
    attachment = store.save(png_bytes(), name="kot.png", chat_id=CHAT)
    agent, provider = build_agent(vision=True, store=store)

    events = list(agent.stream("Opisz obraz", [attachment]))
    assert any(event.get("type") == "token" for event in events)

    user_parts = [m for m in provider.payloads[-1] if m["role"] == "user"][-1]["content"]
    images = [part for part in user_parts if part["type"] == "image_url"]
    texts = [part for part in user_parts if part["type"] == "text"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Opisz obraz" in texts[0]["text"]
    # manifest metadanych (id/nazwa/MIME/wymiary) - bez ścieżki (#51)
    assert attachment.id in texts[0]["text"] and "kot.png" in texts[0]["text"]
    assert "8x6 px" in texts[0]["text"]

    # kontekst agenta: wyłącznie referencje, zero base64 (#51)
    dumped = json.dumps(agent.flow.context)
    assert '"attachment"' in dumped and "base64" not in dumped


def test_no_disk_path_reaches_the_model(store):
    attachment = store.save(png_bytes(), name="zrzut.png", chat_id=CHAT)
    agent, provider = build_agent(vision=True, store=store)
    list(agent.stream("Opisz", [attachment]))
    payload = json.dumps(provider.payloads[-1])
    assert str(store.directory) not in payload
    assert "attachments/" not in payload.split("base64,", 1)[-1]
    assert '"path"' not in payload and '"file"' not in payload


def test_vision_unsupported_blocks_before_provider_call(store):
    attachment = store.save(png_bytes(), chat_id=CHAT)
    agent, provider = build_agent(vision=False, store=store)

    events = list(agent.stream("Opisz", [attachment]))
    assert len(events) == 1 and events[0]["type"] == "error"
    assert "vision" in events[0]["message"]
    assert provider.payloads == [], "obraz nie może polecieć jako plain text"


def test_unknown_capability_is_allowed(store):
    attachment = store.save(png_bytes(), chat_id=CHAT)
    tools = ToolHandle()
    provider = FakeProvider(tools, vision=True)

    provider.supports_vision = lambda: None   # provider nie raportuje capabilities
    agent = OllamaAgent("http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
                        provider=provider, attachment_store=store, chat_id=CHAT)
    events = list(agent.stream("Opisz", [attachment]))
    assert any(event.get("type") == "token" for event in events)
    assert any(part.get("type") == "image_url"
               for part in provider.payloads[-1][-1]["content"])


def test_attachments_without_store_bound_yield_readable_error():
    agent, provider = build_agent(vision=True, store=None)
    events = list(agent.stream("Opisz", ["c" * 32]))
    assert events[0]["type"] == "error" and "sesj" in events[0]["message"]
    assert provider.payloads == []


def test_foreign_chat_attachment_yields_path_free_error(store, tmp_path):
    foreign = store.save(png_bytes(), chat_id="b" * 32)
    agent, provider = build_agent(vision=True, store=store)
    events = list(agent.stream("Opisz", [foreign]))
    assert events[-1]["type"] == "error"
    message = events[-1]["message"]
    assert tmp_path.name not in message and str(store.directory) not in message
    # kontekst wraca do stanu przed wysyłką (snapshot restore)
    assert agent.flow.context[-1]["role"] != "user"


def test_plain_text_message_stays_a_string(store):
    agent, provider = build_agent(vision=True, store=store)
    list(agent.stream("Cześć"))
    assert provider.payloads[-1][-1]["content"] == "Cześć"


def test_large_image_is_downscaled_before_sending(store):
    buffer = io.BytesIO()
    Image.new("RGB", (3000, 2000), "blue").save(buffer, format="PNG")
    attachment = store.save(buffer.getvalue(), name="wielki.png", chat_id=CHAT)
    agent, provider = build_agent(vision=True, store=store)
    list(agent.stream("Opisz", [attachment]))

    url = [part for part in provider.payloads[-1][-1]["content"]
           if part["type"] == "image_url"][0]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
        assert max(image.size) <= SEND_MAX_SIDE


# ---------------------------------------------------------- token estimate (C)
def test_attachment_parts_are_counted_in_token_estimate():
    multimodal = {"role": "user", "content": [{"type": "text", "text": "opisz"},
                                             {"type": "attachment", "id": "a" * 32}]}
    tokens = ContextHandle.MessageTokens(multimodal)
    assert ATTACHMENT_TOKENS <= tokens < ATTACHMENT_TOKENS + 20
    # data URL po resolve() kosztuje tyle samo dla budżetu (#51)
    resolved = {"role": "user", "content": [{"type": "image_url",
                                            "image_url": {"url": "data:image/png;base64," + "A" * 1_000_000}}]}
    assert ContextHandle.MessageTokens(resolved) == ATTACHMENT_TOKENS + 5


def test_plain_message_estimate_is_unchanged():
    assert ContextHandle.MessageTokens({"role": "user", "content": "abcd"}) == 6
