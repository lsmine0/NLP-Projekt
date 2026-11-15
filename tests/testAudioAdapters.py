from __future__ import annotations

from pathlib import Path

import pytest

from core import AgentAdapter
from core.AgentAdapter import RealSTT, RealTTS, StubSTT, StubTTS, create_stt, create_tts


class FakeSTTHandle:
    def __init__(self) -> None:
        self.started = False

    def start_recording(self) -> None:
        self.started = True

    def stop_recording(self) -> str:
        return "rozpoznany tekst"


class FakeTTSHandle:
    def __init__(self, last_error: Exception | None = None) -> None:
        self.last_error = last_error
        self.calls: list[tuple[str, object]] = []

    def speak_stream(self, tokens: tuple[str, ...]) -> None:
        self.calls.append(("speak_stream", tokens))

    def wait(self) -> bool:
        self.calls.append(("wait", None))
        return True

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def close(self) -> None:
        self.calls.append(("close", None))


def test_real_stt_delegates_recording() -> None:
    handle = FakeSTTHandle()
    stt = RealSTT(handle)  # type: ignore[arg-type]

    stt.start()

    assert handle.started
    assert stt.stop() == "rozpoznany tekst"


def test_real_tts_streams_waits_and_propagates_playback_error() -> None:
    playback_error = RuntimeError("brak modelu Piper")
    handle = FakeTTSHandle(playback_error)
    tts = RealTTS(handle)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="brak modelu Piper"):
        tts.speak("Odpowiedź agenta.")

    assert handle.calls == [
        ("speak_stream", ("Odpowiedź agenta.",)),
        ("wait", None),
    ]

    tts.stop()
    tts.close()
    assert handle.calls[-2:] == [("stop", None), ("close", None)]


def test_create_stt_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AgentAdapter, "_missing_audio_dependencies", lambda *_: [])
    stt = create_stt(
        {
            "stt_model": "small",
            "stt_device": "cpu",
            "stt_input_device": 3,
            "stt_language": "en",
        }
    )

    assert isinstance(stt, RealSTT)
    assert stt.handle.model_size == "small"
    assert stt.handle.device == "cpu"
    assert stt.handle.input_device == 3
    assert stt.handle.language == "en"


def test_create_tts_uses_model_and_sink_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(AgentAdapter, "_missing_audio_dependencies", lambda *_: [])
    model_path = tmp_path / "custom.onnx"
    tts = create_tts({"tts_model": str(model_path), "tts_sink_device": 4})

    assert isinstance(tts, RealTTS)
    assert tts.handle.model_path == model_path
    assert tts.handle._sink.device == 4


def test_audio_factories_fall_back_when_dependencies_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AgentAdapter, "_missing_audio_dependencies", lambda *_: ["sounddevice"]
    )

    assert isinstance(create_stt(), StubSTT)
    assert isinstance(create_tts(), StubTTS)
