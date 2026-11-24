from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Cli as app_cli
from core import main as app_main
from core.AgentAdapter import AgentBase
from core.Attachments import Attachment, AttachmentStore
from core.ChatStore import ChatStore, SettingsStore, format_chat_export, safe_export_filename
from core.Interactions import InteractionBroker
from handlers.ToolHandle import ToolHandle


def _png_bytes(color: str = "navy") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def test_export_empty_conversation_txt_and_md(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats")
    chat = store.create(title="Pusta rozmowa testowa")

    md = store.export_chat(chat["id"], "md")
    txt = store.export_chat(chat["id"], "txt")

    assert md.format == "md"
    assert md.media_type == "text/markdown; charset=utf-8"
    assert md.filename.startswith("pusta-rozmowa-testowa-")
    assert md.filename.endswith(".md")
    assert "# Pusta rozmowa testowa" in md.content
    assert "Brak wiadomości w rozmowie." in md.content

    assert txt.format == "txt"
    assert txt.media_type == "text/plain; charset=utf-8"
    assert txt.filename.endswith(".txt")
    assert "Pusta rozmowa testowa" in txt.content
    assert "Brak wiadomości w rozmowie." in txt.content


def test_export_preserves_order_roles_polish_chars_code_and_mermaid(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats")
    chat = store.create(title="Zażółć gęślą jaźń — kod i diagram")
    chat_id = chat["id"]

    user_1 = "Cześć! Pokaż funkcję w Pythonie z polskimi znakami:\nąćęłńóśźż."
    assistant_1 = (
        "Oczywiście, już się tym zajmuję.\n\n"
        "```python\n"
        "def przywitaj(imie: str) -> str:\n"
        "    return f'Cześć, {imie}!'\n"
        "```\n\n"
        "Poniżej diagram przepływu:\n\n"
        "```mermaid\n"
        "flowchart TD\n"
        "    A[Start] --> B[Koniec]\n"
        "```"
    )
    user_2 = "Dzięki!\nDruga linia pytania."
    assistant_2 = "Do usług, Panie!"

    store.append(chat_id, "user", user_1)
    store.append(chat_id, "assistant", assistant_1)
    store.append(chat_id, "user", user_2)
    store.append(chat_id, "assistant", assistant_2)

    for fmt in ("md", "txt"):
        exported = store.export_chat(chat_id, fmt)
        content = exported.content
        assert "Zażółć gęślą jaźń — kod i diagram" in content
        assert "ąćęłńóśźż" in content
        assert "```python\ndef przywitaj(imie: str) -> str:" in content
        assert "```mermaid\nflowchart TD\n    A[Start] --> B[Koniec]\n```" in content
        assert "Dzięki!\nDruga linia pytania." in content

        idx_u1 = content.index("Cześć! Pokaż funkcję")
        idx_a1 = content.index("Oczywiście, już się tym zajmuję.")
        idx_u2 = content.index("Dzięki!\nDruga linia pytania.")
        idx_a2 = content.index("Do usług, Panie!")
        assert idx_u1 < idx_a1 < idx_u2 < idx_a2

        if fmt == "md":
            assert "## Użytkownik" in content
            assert "## Asystent" in content
        else:
            assert "[Użytkownik]" in content
            assert "[Asystent]" in content


def test_export_strips_thinking_and_formats_interrupted_responses(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats")
    chat = store.create(title="Test znaczników technicznych")
    chat_id = chat["id"]

    store.append(chat_id, "user", "Pierwsze pytanie")
    store.append(
        chat_id,
        "assistant",
        "<think>Tajny proces myślenia\nw dwóch liniach</think>\nWidoczna odpowiedź.",
    )
    store.append(chat_id, "user", "Drugie pytanie")
    store.append(
        chat_id,
        "assistant",
        "<think>Nieukończone myślenie</think>\nCzęściowa odpowiedź\n<stopped/>",
    )
    store.append(chat_id, "user", "Trzecie pytanie")
    store.append(chat_id, "assistant", "<think>Tylko myślenie")
    store.append(chat_id, "user", "Czwarte pytanie")
    store.append(chat_id, "assistant", "<stopped/>")

    md = store.export_chat(chat_id, "md").content
    txt = store.export_chat(chat_id, "txt").content

    for content in (md, txt):
        assert "<think" not in content.lower()
        assert "</think" not in content.lower()
        assert "<stopped" not in content.lower()
        assert "Tajny proces myślenia" not in content
        assert "Nieukończone myślenie" not in content
        assert "Tylko myślenie" not in content
        assert "Widoczna odpowiedź." in content
        assert "Częściowa odpowiedź" in content
        assert "Odpowiedź została przerwana przez użytkownika" in content
        assert "(brak treści odpowiedzi)" in content


def test_export_attachments_without_paths_base64_or_secrets(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats")
    att_store = AttachmentStore(tmp_path / "attachments")
    chat = store.create(title="Rozmowa z załącznikiem")
    chat_id = chat["id"]

    saved = att_store.save(_png_bytes(), name=r"C:\sekretny\folder\wykres_sprzedaży.png", chat_id=chat_id)
    missing_id = "f" * 32

    store.append(chat_id, "user", "Przeanalizuj ten obraz.", attachments=[saved.id, missing_id])
    store.append(
        chat_id,
        "assistant",
        "Na wykresie widać wzrost.",
        tools=[{"name": "inspect_image", "source": "builtin", "risk": ["read"], "status": "ok"}],
    )

    for fmt in ("md", "txt"):
        exported = store.export_chat(chat_id, fmt, attachment_store=att_store)
        content = exported.content
        assert "wykres_sprzedaży.png" in content
        assert "image/png" in content
        assert saved.id in content
        assert missing_id in content
        assert str(tmp_path) not in content
        assert "sekretny" not in content
        assert "data:image" not in content
        assert "iVBORw0KGgo" not in content
        assert "inspect_image" not in content
        assert "ollama_endpoint" not in content
        assert "api_key" not in content


def test_export_safe_filename_and_errors(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats")
    chat = store.create(title='../../Zły: "Tytuł"? <test> łódź')

    name_md = safe_export_filename(chat, "MD")
    assert name_md.endswith(".md")
    assert "/" not in name_md and "\\" not in name_md
    assert ".." not in name_md
    assert "lodz" in name_md
    assert chat["id"][:8] in name_md

    with pytest.raises(ValueError, match="Nieobsługiwany format"):
        store.export_chat(chat["id"], "pdf")
    with pytest.raises(ValueError, match="Nieobsługiwany format"):
        store.export_chat(chat["id"], "")
    with pytest.raises(KeyError):
        store.export_chat("nie-istnieje", "txt")
    with pytest.raises(KeyError):
        store.export_chat("../escape", "md")


def test_export_http_response_contract(tmp_path: Path) -> None:
    from urllib.parse import quote

    store = ChatStore(tmp_path / "chats")
    att_store = AttachmentStore(tmp_path / "attachments")
    chat = store.create(title="Eksport HTTP żźć")
    store.append(chat["id"], "user", "Hej, sprawdź kod:\n```js\nconsole.log('ąę');\n```")
    store.append(chat["id"], "assistant", "<think>ok</think>Działa poprawnie!")

    api = FastAPI()

    @api.get("/api/chats/{chat_id}/export")
    def export_route(chat_id: str, format: str = "md"):
        try:
            exported = store.export_chat(chat_id, format, attachment_store=att_store)
        except ValueError as exc:
            return JSONResponse({"error": str(exc), "code": "invalid_format"}, status_code=400)
        except KeyError:
            return JSONResponse({"error": "Nie znaleziono czatu."}, status_code=404)
        return Response(
            content=exported.content.encode("utf-8"),
            media_type=exported.media_type,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{exported.filename}"; '
                    f"filename*=UTF-8''{quote(exported.filename)}"
                ),
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    with TestClient(api) as client:
        res_md = client.get(f"/api/chats/{chat['id']}/export?format=md")
        assert res_md.status_code == 200
        assert res_md.headers["content-type"] == "text/markdown; charset=utf-8"
        assert res_md.headers["content-disposition"].startswith("attachment; filename=")
        assert ".md" in res_md.headers["content-disposition"]
        text_md = res_md.content.decode("utf-8")
        assert "Działa poprawnie!" in text_md
        assert "<think>" not in text_md

        res_txt = client.get(f"/api/chats/{chat['id']}/export?format=txt")
        assert res_txt.status_code == 200
        assert res_txt.headers["content-type"] == "text/plain; charset=utf-8"
        assert ".txt" in res_txt.headers["content-disposition"]
        assert "console.log('ąę');" in res_txt.content.decode("utf-8")

        res_bad = client.get(f"/api/chats/{chat['id']}/export?format=pdf")
        assert res_bad.status_code == 400
        assert res_bad.json()["code"] == "invalid_format"

        res_404 = client.get("/api/chats/missing-id/export?format=txt")
        assert res_404.status_code == 404


def test_webgui_export_endpoint_when_multipart_available(tmp_path: Path) -> None:
    try:
        from core.WebGui import create_app
        from core.main import build_api_runtime
        from core.RuntimeConfig import RuntimeConfig
    except RuntimeError as exc:
        if "python-multipart" in str(exc):
            pytest.skip("python-multipart nie jest zainstalowany w tym środowisku")
        raise

    config = RuntimeConfig(data_dir=tmp_path / "data", workspace=tmp_path, model="test-model")
    runtime = build_api_runtime(
        config,
        provider_factory=lambda cfg, tools: NS(
            config=cfg, ToolsHandle=tools, base_url=cfg.endpoint, model="test-model", close=lambda: None
        ),
        tool_factory=ToolHandle,
    )
    app = create_app(runtime=runtime)
    try:
        chat = runtime.chat_store.create(title="Integracja WebGui")
        runtime.chat_store.append(chat["id"], "user", "Pytanie")
        runtime.chat_store.append(chat["id"], "assistant", "Odpowiedź")

        with TestClient(app) as client:
            res = client.get(f"/api/chats/{chat['id']}/export?format=txt")
            assert res.status_code == 200
            assert res.headers["content-type"] == "text/plain; charset=utf-8"
            assert "Odpowiedź" in res.content.decode("utf-8")
            assert client.get(f"/api/chats/{chat['id']}/export?format=zip").status_code == 400
            assert client.get("/api/chats/brak/export?format=md").status_code == 404
    finally:
        runtime.close()


def test_cli_export_command_online_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class EchoAgent(AgentBase):
        def reset(self) -> None:
            pass

        def load_history(self, messages: list[dict]) -> None:
            pass

        def stream(
            self, user_text: str, attachments: list[Attachment] | None = None
        ) -> Iterator[dict]:
            yield {"type": "thinking", "text": "ukryte"}
            yield {"type": "token", "text": "Odpowiedź CLI z ąęś."}

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    runtime = app_main.AppRuntime(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        chat_store=ChatStore(tmp_path / "chats"),
        tool_handle=ToolHandle(),
        provider=NS(
            model="test-model",
            base_url="http://127.0.0.1:11434/v1",
            GetModels=lambda: [NS(id="test-model")],
            SetModel=lambda _n: True,
        ),  # type: ignore[arg-type]
        stt=app_main.DisabledSTT(),
        tts=app_main.DisabledTTS(),
        agent_factory=lambda _s: EchoAgent(),
        attachment_store=AttachmentStore(tmp_path / "attachments"),
        audio_enabled=False,
        interaction_broker=InteractionBroker(),
    )

    chosen_md = tmp_path / "chosen" / "rozmowa.md"
    custom_txt = tmp_path / "exports" / "moj_czat.txt"
    inputs = iter([
        "/export",
        "Pytanie do eksportu",
        "/export",
        "",
        str(chosen_md),
        "/export txt",
        "anuluj",
        f"/export txt {custom_txt}",
        "/export pdf",
        "exit",
    ])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))

    try:
        assert app_main.run_cli(runtime) == 0
        out = capsys.readouterr().out
        assert "Brak aktywnej zapisanej rozmowy" in out
        assert "Anulowano eksport." in out
        assert "Nieobsługiwany format eksportu" in out
        assert "Wyeksportowano rozmowę do:" in out

        assert list(work_dir.glob("*")) == []
        assert chosen_md.is_file()
        md_text = chosen_md.read_text(encoding="utf-8")
        assert "Pytanie do eksportu" in md_text
        assert "Odpowiedź CLI z ąęś." in md_text
        assert "<think>" not in md_text

        assert custom_txt.is_file()
        txt_text = custom_txt.read_text(encoding="utf-8")
        assert "[Użytkownik]" in txt_text
        assert "Odpowiedź CLI z ąęś." in txt_text

        # Eksport po /load w trybie offline
        runtime.provider_warning = "Brak połączenia"
        offline_dir = tmp_path / "offline_out"
        offline_dir.mkdir()
        inputs2 = iter(["/load 1", f"/export md {offline_dir}", "exit"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs2))
        assert app_main.run_cli(runtime) == 0
        offline_md = list(offline_dir.glob("*.md"))
        assert len(offline_md) == 1
        assert "Odpowiedź CLI z ąęś." in offline_md[0].read_text(encoding="utf-8")
    finally:
        runtime.close()


def test_frontend_contains_export_controls_and_endpoint_calls() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="export-toggle"' in html
    assert 'id="export-toggle" type="button" disabled' in html
    assert 'id="export-dropdown"' in html
    assert 'id="export-md"' in html
    assert 'id="export-txt"' in html
    assert "function updateExportState()" in html
    assert "async function exportActiveChat(format)" in html
    assert "/export?format=" in html
    assert "window.pywebview?.api?.save_export_file" in html
    assert "showSaveFilePicker" in html


def test_desktop_gui_api_save_export_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from core.WebGui import _DesktopGuiApi
    except RuntimeError as exc:
        if "python-multipart" in str(exc):
            pytest.skip("python-multipart nie jest zainstalowany w tym środowisku")
        raise

    import types

    fake_webview = types.SimpleNamespace(
        FileDialog=types.SimpleNamespace(SAVE=20),
        SAVE_DIALOG=20,
    )
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    target = tmp_path / "eksport_zażółć.md"

    class FakeWindow:
        def __init__(self, chosen) -> None:
            self.chosen = chosen
            self.calls = []

        def create_file_dialog(self, dialog_type, **kwargs):
            self.calls.append((dialog_type, kwargs))
            return self.chosen

    api = _DesktopGuiApi()
    assert not hasattr(api, "window")
    api._window = FakeWindow((str(target),))
    res = api.save_export_file("rozmowa.md", "# Tytuł\nzażółć", "md")
    assert res == {"saved": True, "path": str(target)}
    assert target.read_text(encoding="utf-8") == "# Tytuł\nzażółć"

    api._window = FakeWindow(None)
    assert api.save_export_file("rozmowa.txt", "abc", "txt") == {"saved": False, "cancelled": True}


def test_cli_export_uses_native_save_dialog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class EchoAgent(AgentBase):
        def reset(self) -> None:
            pass

        def load_history(self, messages: list[dict]) -> None:
            pass

        def stream(
            self, user_text: str, attachments: list[Attachment] | None = None
        ) -> Iterator[dict]:
            yield {"type": "token", "text": "Odpowiedź z dialogiem."}

    runtime = app_main.AppRuntime(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        chat_store=ChatStore(tmp_path / "chats"),
        tool_handle=ToolHandle(),
        provider=NS(
            model="test-model",
            base_url="http://127.0.0.1:11434/v1",
            GetModels=lambda: [NS(id="test-model")],
            SetModel=lambda _n: True,
        ),  # type: ignore[arg-type]
        stt=app_main.DisabledSTT(),
        tts=app_main.DisabledTTS(),
        agent_factory=lambda _s: EchoAgent(),
        attachment_store=AttachmentStore(tmp_path / "attachments"),
        audio_enabled=False,
        interaction_broker=InteractionBroker(),
    )

    chosen_path = tmp_path / "dialog_out" / "wybrany.md"
    dialog_results = iter([(True, chosen_path), (True, None)])
    dialog_calls: list[tuple[str, str, Path]] = []

    def fake_dialog(filename: str, fmt: str, initial_dir: Path):
        dialog_calls.append((filename, fmt, initial_dir))
        return next(dialog_results)

    monkeypatch.setattr(app_cli, "_cli_save_file_dialog", fake_dialog)
    inputs = iter(["Pytanie", "/export md", "/export txt", "exit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))

    try:
        assert app_main.run_cli(runtime) == 0
        out = capsys.readouterr().out
        assert len(dialog_calls) == 2
        assert dialog_calls[0][1] == "md"
        assert dialog_calls[1][1] == "txt"
        assert chosen_path.is_file()
        assert "Odpowiedź z dialogiem." in chosen_path.read_text(encoding="utf-8")
        assert "Anulowano eksport." in out
    finally:
        runtime.close()
