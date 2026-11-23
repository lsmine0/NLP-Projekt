from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import main as app_main
from core.AgentAdapter import AgentBase, RealTTS, TTSBase
from core.Attachments import Attachment, AttachmentStore
from core.ChatStore import ChatStore, SettingsStore
from core.Interactions import InteractionBroker
from handlers.OllamaHandle import OllamaHandler
from handlers.ToolHandle import ToolHandle
from handlers.TTSHandle import TTSHandle


class DummyAgent(AgentBase):
    def __init__(self) -> None:
        self.loaded_histories: list[list[dict]] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def load_history(self, messages: list[dict]) -> None:
        self.loaded_histories.append(list(messages))

    def stream(
        self, user_text: str, attachments: list[Attachment] | None = None
    ) -> Iterator[dict]:
        yield {"type": "token", "text": f"Echo: {user_text}"}


def _make_runtime(
    tmp_path: Path,
    *,
    provider: Any = None,
    agent_factory: Any = None,
    tts: TTSBase | None = None,
    audio_enabled: bool = False,
    provider_warning: str | None = None,
    settings_updates: dict[str, Any] | None = None,
) -> app_main.AppRuntime:
    settings_store = SettingsStore(tmp_path / "settings.json")
    if settings_updates:
        settings_store.update(settings_updates)
    chat_store = ChatStore(tmp_path / "chats")
    attachment_store = AttachmentStore(tmp_path / "attachments")
    tool_handle = ToolHandle()
    broker = InteractionBroker()
    if provider is None:
        provider = NS(
            model="test-model",
            base_url="http://127.0.0.1:11434/v1",
            GetModels=lambda: [NS(id="test-model")],
            SetModel=lambda name: True,
        )
    if tts is None:
        tts = app_main.DisabledTTS()
    if agent_factory is None:
        agent_factory = lambda _settings: DummyAgent()

    return app_main.AppRuntime(
        settings_store=settings_store,
        chat_store=chat_store,
        tool_handle=tool_handle,
        provider=provider,
        stt=app_main.DisabledSTT(),
        tts=tts,
        agent_factory=agent_factory,
        attachment_store=attachment_store,
        audio_enabled=audio_enabled,
        interaction_broker=broker,
        provider_warning=provider_warning,
    )


def _feed_inputs(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    iterator = iter(values)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", fake_input)


def test_build_runtime_cli_offline_vs_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "settings.json"
    config_file.write_text('{"ollama_model": "qwen3.5:4b"}', encoding="utf-8")

    monkeypatch.setattr(app_main, "build_tools", lambda **_kwargs: ToolHandle())

    def fail_get_models(self: Any) -> list[Any]:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(OllamaHandler, "GetModels", fail_get_models)

    cli_args, _ = app_main.parse_args(
        ["--cli", "--no-audio", "--no-mcp", "--config", str(config_file)]
    )
    runtime = app_main.build_runtime(cli_args)
    try:
        assert runtime.provider_warning is not None
        assert "Brak połączenia" in runtime.provider_warning
        assert runtime.provider.model is None
    finally:
        runtime.close()

    bench_args, _ = app_main.parse_args(
        ["--benchmark", "--no-audio", "--no-mcp", "--config", str(config_file)]
    )
    with pytest.raises(app_main.StartupError, match="Brak połączenia"):
        app_main.build_runtime(bench_args)


def test_cli_offline_mode_commands_and_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    online = False

    class FlakyProvider:
        def __init__(self) -> None:
            self.model: str | None = None
            self.base_url = "http://127.0.0.1:11434/v1"

        def GetModels(self) -> list[Any]:
            if not online:
                raise ConnectionError("offline")
            return [NS(id="qwen3.5:4b")]

        def SetModel(self, name: str) -> bool:
            self.model = name
            return True

    class ScriptedAgent(AgentBase):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.model = ""

        def reset(self) -> None:
            pass

        def load_history(self, messages: list[dict]) -> None:
            pass

        def stream(
            self, user_text: str, attachments: list[Attachment] | None = None
        ) -> Iterator[dict]:
            self.calls.append(user_text)
            yield {"type": "token", "text": "Połączono i odpowiedziano."}

    agent = ScriptedAgent()
    runtime = _make_runtime(
        tmp_path,
        provider=FlakyProvider(),
        agent_factory=lambda _s: agent,
        provider_warning="Brak połączenia z serwerem modeli (http://127.0.0.1:11434/v1).",
    )
    try:
        _feed_inputs(monkeypatch, ["/chats", "/logs", "/reset", "Cześć offline", "exit"])
        assert app_main.run_cli(runtime) == 0
        out = capsys.readouterr().out
        assert "(offline)" in out
        assert "Tryb offline" in out
        assert "Brak zapisanych rozmów" in out
        assert agent.calls == []
        assert runtime.chat_store.list() == []

        # Ollama uruchomiona później — kolejna wiadomość w tej samej sesji łączy się automatycznie.
        online = True
        _feed_inputs(monkeypatch, ["Hej ponownie", "exit"])
        assert app_main.run_cli(runtime) == 0
        out2 = capsys.readouterr().out
        assert "Połączono z serwerem modeli (qwen3.5:4b)" in out2
        assert agent.calls == ["Hej ponownie"]
        assert agent.model == "qwen3.5:4b"
        assert len(runtime.chat_store.list()) == 1
    finally:
        runtime.close()


def test_cli_persists_chat_history_reset_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class RichAgent(AgentBase):
        def __init__(self) -> None:
            self.loaded: list[list[dict]] = []
            self.reset_count = 0
            self.turn = 0

        def reset(self) -> None:
            self.reset_count += 1

        def load_history(self, messages: list[dict]) -> None:
            self.loaded.append(list(messages))

        def stream(
            self, user_text: str, attachments: list[Attachment] | None = None
        ) -> Iterator[dict]:
            self.turn += 1
            if self.turn == 1:
                yield {"type": "token", "text": "brudnopis"}
                yield {"type": "answer_reset", "reason": "tool_json"}
                yield {"type": "thinking", "text": "Analizuję plik."}
                yield {"type": "token", "text": "Gotowa odpowiedź 1"}
            else:
                yield {"type": "token", "text": f"Odpowiedź na: {user_text}"}

    agent = RichAgent()
    runtime = _make_runtime(tmp_path, agent_factory=lambda _s: agent)
    try:
        _feed_inputs(
            monkeypatch,
            [
                "Pierwsze pytanie",
                "/reset",
                "Drugie pytanie",
                "/chats",
                "/load",
                "/load brak",
                "/load 2",
                "Kontynuacja pierwszego",
                "exit",
            ],
        )
        assert app_main.run_cli(runtime) == 0
        out = capsys.readouterr().out
        assert "Użycie: /load" in out
        assert "Nie znaleziono czatu 'brak'" in out
        assert "Wczytano rozmowę" in out

        chats = runtime.chat_store.list()
        assert len(chats) == 2
        first_chat = next(
            c
            for c in (runtime.chat_store.get(item["id"]) for item in chats)
            if c["title"].startswith("Pierwsze")
        )
        second_chat = next(
            c
            for c in (runtime.chat_store.get(item["id"]) for item in chats)
            if c["title"].startswith("Drugie")
        )

        assert len(second_chat["messages"]) == 2
        assert len(first_chat["messages"]) == 4
        assistant_msg = first_chat["messages"][1]
        assert "brudnopis" not in assistant_msg["content"]
        assert "<think>Analizuję plik.</think>" in assistant_msg["content"]
        assert "Gotowa odpowiedź 1" in assistant_msg["content"]
        assert "tools" not in assistant_msg
        assert "tool_calls" not in assistant_msg
        assert len(agent.loaded) == 1
        assert len(agent.loaded[0]) == 2

        # Sprawdź wznawianie po prefiksie ID w nowej sesji CLI
        prefix = second_chat["id"][:8]
        _feed_inputs(monkeypatch, [f"/load {prefix}", "exit"])
        assert app_main.run_cli(runtime) == 0
        assert len(agent.loaded) == 2
        assert agent.loaded[1][0]["content"] == "Drugie pytanie"
    finally:
        runtime.close()


def test_cli_tts_status_and_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[tuple[str, str]] = []

    class InterruptibleTTS(TTSBase):
        def __init__(self, interrupt: bool = True) -> None:
            self.interrupt = interrupt

        def speak(self, text: str) -> None:
            events.append(("speak", text))
            if self.interrupt:
                raise KeyboardInterrupt

        def stop(self) -> None:
            events.append(("stop", ""))

    runtime = _make_runtime(
        tmp_path,
        tts=InterruptibleTTS(interrupt=True),
        audio_enabled=True,
        settings_updates={"tts_enabled": True},
    )
    try:
        _feed_inputs(
            monkeypatch,
            [
                "Powiedz coś",
                "/tts test",
                "/tts własny tekst",
                "exit",
            ],
        )
        assert app_main.run_cli(runtime) == 0
        out = capsys.readouterr().out
        assert "[TTS] Odtwarzanie odpowiedzi..." in out
        assert "[TTS] Testowanie syntezy mowy:" in out
        assert "[TTS] Odtwarzanie tekstu:" in out
        assert out.count("[TTS] Odtwarzanie przerwane.") == 3
        assert events == [
            ("speak", "Echo: Powiedz coś"),
            ("stop", ""),
            ("speak", "Cześć! Synteza mowy Piper działa poprawnie."),
            ("stop", ""),
            ("speak", "własny tekst"),
            ("stop", ""),
        ]
    finally:
        runtime.close()


def test_real_tts_and_tts_handle_wait_interrupt_cleanup() -> None:
    calls: list[str] = []

    class FakeHandle:
        last_error = None

        def speak_stream(self, tokens: Any) -> None:
            calls.append("speak_stream")

        def wait(self) -> bool:
            calls.append("wait")
            raise KeyboardInterrupt

        def stop(self) -> None:
            calls.append("stop")

    tts = RealTTS(FakeHandle())  # type: ignore[arg-type]
    with pytest.raises(KeyboardInterrupt):
        tts.speak("Test")
    assert calls == ["speak_stream", "wait", "stop"]

    # TTSHandle.wait() poprawnie kończy po zakończeniu wątków i respektuje timeout
    handle = TTSHandle(auto_download=False)
    done = threading.Event()

    def worker() -> None:
        time.sleep(0.05)
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    handle._threads = [thread]
    assert handle.wait(timeout=0.01) is False
    assert handle.wait(timeout=1.0) is True
    assert done.is_set()
