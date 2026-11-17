from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import ChatStore as chat_store_module
from core import WebGui
from core import main as app_main
from handlers.OllamaHandle import OllamaHandler
from handlers.ToolHandle import ToolHandle


def test_parse_args_modes_and_flags() -> None:
    default_args, default_extra = app_main.parse_args([])
    assert default_args.gui is True
    assert default_args.cli is False
    assert default_args.benchmark is False
    assert default_args.provider == "ollama"
    assert default_extra == []

    cli_args, cli_extra = app_main.parse_args(
        ["--cli", "--no-audio", "--no-mcp", "--provider", "local"]
    )
    assert cli_args.cli is True
    assert cli_args.gui is False
    assert cli_args.no_audio is True
    assert cli_args.no_mcp is True
    assert cli_args.provider == "local"
    assert cli_extra == []

    cfg_args, _ = app_main.parse_args(["--config"])
    assert cfg_args.config == "__show__"

    bench_args, bench_extra = app_main.parse_args(["--benchmark", "--ids", "B01", "B02"])
    assert bench_args.benchmark is True
    assert bench_args.gui is False
    assert bench_extra == ["--ids", "B01", "B02"]


def test_build_config_file_and_inline_updates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_store_module, "DATA_DIR", tmp_path)

    settings_store, chat_store = app_main.build_config("tts_enabled=false")
    assert settings_store.path == tmp_path / "settings.json"
    assert chat_store.directory == tmp_path / "chats"
    assert settings_store.get()["tts_enabled"] is False

    settings_store, _ = app_main.build_config("max_chats=25")
    assert settings_store.get()["max_chats"] == 25

    custom_path = tmp_path / "custom" / "my_settings.json"
    custom_store, custom_chats = app_main.build_config(str(custom_path))
    assert custom_store.path == custom_path.resolve()
    assert custom_chats.directory == custom_path.resolve().parent / "chats"

    for invalid_arg in ("nieznany=1", "tts_enabled=moze", "max_chats=abc"):
        with pytest.raises(app_main.StartupError):
            app_main.build_config(invalid_arg)


def test_main_build_tools_registers_all_tools() -> None:
    tools = app_main.build_tools()
    assert isinstance(tools, ToolHandle)
    assert len(tools.get_tools_list()) == 18


def test_build_provider_startup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = ToolHandle()
    settings = {"ollama_endpoint": "http://127.0.0.1:11434/v1", "ollama_model": ""}

    with pytest.raises(app_main.StartupError, match="Nieobsługiwany provider"):
        app_main.build_provider(settings, tool_handle=tools, provider_name="nieznany")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(app_main.StartupError, match="Brak klucza API"):
        app_main.build_provider(settings, tool_handle=tools, provider_name="openai")

    def fail_get_models(self):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(OllamaHandler, "GetModels", fail_get_models)

    with pytest.raises(app_main.StartupError, match="Brak połączenia"):
        app_main.build_provider(
            settings, tool_handle=tools, require_connection=True
        )

    handler = app_main.build_provider(
        settings, tool_handle=tools, require_connection=False
    )
    assert isinstance(handler, OllamaHandler)
    assert handler.model is None


def test_build_audio_no_audio_returns_disabled_adapters() -> None:
    stt, tts = app_main.build_audio(no_audio=True)
    assert isinstance(stt, app_main.DisabledSTT)
    assert isinstance(tts, app_main.DisabledTTS)

    with pytest.raises(RuntimeError):
        stt.start()
    with pytest.raises(RuntimeError):
        tts.speak("test")


def test_build_runtime_and_close_releases_resources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "settings.json"
    config_file.write_text('{"ollama_model": "qwen3.5:4b"}', encoding="utf-8")

    monkeypatch.setattr(
        OllamaHandler,
        "GetModels",
        lambda self: [SimpleNamespace(id="qwen3.5:4b")],
    )
    monkeypatch.setattr(
        OllamaHandler,
        "SetModel",
        lambda self, name: setattr(self, "model", name) is None or True,
    )

    args, _ = app_main.parse_args(
        ["--cli", "--no-audio", "--no-mcp", "--config", str(config_file)]
    )
    runtime = app_main.build_runtime(args)

    assert runtime.provider.model == "qwen3.5:4b"
    assert runtime.audio_enabled is False
    assert isinstance(runtime.stt, app_main.DisabledSTT)
    assert isinstance(runtime.tts, app_main.DisabledTTS)

    closed_events: list[str] = []

    class FakeAudio:
        def __init__(self, label: str) -> None:
            self.label = label

        def close(self) -> None:
            closed_events.append(f"audio:{self.label}")

    class FakeMCP:
        def close(self) -> None:
            closed_events.append("mcp")

    runtime.tts = FakeAudio("tts")  # type: ignore[assignment]
    runtime.stt = FakeAudio("stt")  # type: ignore[assignment]
    runtime.mcp_clients.append(FakeMCP())

    WebGui.sessions["test-chat"] = object()  # type: ignore[assignment]
    try:
        runtime.close()
        assert closed_events == ["audio:tts", "audio:stt", "mcp"]
        assert runtime.mcp_clients == []
        assert WebGui.sessions == {}
    finally:
        WebGui.sessions.clear()


def test_webgui_exposes_start_gui_without_side_effects_on_import() -> None:
    assert callable(WebGui.start_gui)
    assert WebGui.app is not None


def test_put_settings_updates_system_prompt_and_rejects_non_text(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_store = chat_store_module.SettingsStore(tmp_path / "settings.json")
    chat_store = chat_store_module.ChatStore(directory=tmp_path / "chats")
    monkeypatch.setattr(WebGui, "settings_store", settings_store)
    monkeypatch.setattr(WebGui, "store", chat_store)
    WebGui.sessions["existing"] = object()  # type: ignore[assignment]

    try:
        updated = WebGui.put_settings({"system_prompt": "Instrukcje użytkownika"})
        assert isinstance(updated, dict)
        assert updated["system_prompt"] == "Instrukcje użytkownika"
        assert WebGui.sessions == {}

        invalid = WebGui.put_settings({"system_prompt": ["nie", "tekst"]})
        assert not isinstance(invalid, dict)
        assert invalid.status_code == 400
    finally:
        WebGui.sessions.clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
