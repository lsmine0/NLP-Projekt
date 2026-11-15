from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Iterator
from core import Logger
from core.AgentFlow import AgentFlow
from handlers.ContextHandle import ContextHandle
from handlers.OllamaHandle import OllamaHandler
from handlers.STTHandle import STTHandle
from handlers.TTSHandle import SoundDeviceSink, TTSHandle


# Interfejs bazowy agenta dla czatu
class AgentBase:

    def stream(self, user_text: str) -> Iterator[dict]:
        raise NotImplementedError

    def run(self, user_text: str) -> str:
        parts = []
        for event in self.stream(user_text):
            if event["type"] == "token":
                parts.append(event["text"])
            elif event["type"] == "error":
                raise RuntimeError(event["message"])
        return "".join(parts)

    def reset(self) -> None:
        raise NotImplementedError

    def load_history(self, messages: list[dict]) -> None:
        raise NotImplementedError


# Czat z modelem oparty na AgentFlow (wspólna pętla z toolami i budżetem kontekstu)
class OllamaAgent(AgentBase):
    default_tool_handle: Any = None
    default_system_prompt: str | None = None
    default_api_key: str = "ollam"

    def __init__(
        self,
        endpoint: str,
        model: str = "",
        system_prompt: str | None = None,
        tool_handle: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.system_prompt = (
            system_prompt if system_prompt is not None else self.default_system_prompt
        )
        self.tool_handle = (
            tool_handle if tool_handle is not None else self.default_tool_handle
        )
        self.api_key = api_key or self.default_api_key
        self.flow = AgentFlow(
            base_url=self.endpoint,
            key=self.api_key,
            system_prompt=self.system_prompt,
        )
        if self.tool_handle is not None:
            self.flow.ollama.ToolsHandle = self.tool_handle
            self.flow.tools = self.tool_handle
        self.handler: OllamaHandler = self.flow.ollama
        self.history: list[dict] = self.flow.context

    def reset(self) -> None:
        self.flow.context.clear()
        if self.system_prompt:
            self.flow.context.append(ContextHandle.Create(self.system_prompt))
        self.history = self.flow.context

    def load_history(self, messages: list[dict]) -> None:
        self.reset()
        self.flow.context.extend(
            ContextHandle.Create(message["content"], message["role"])
            for message in messages
            if message.get("role") in ("user", "assistant")
        )
        self.history = self.flow.context

    def _get_handler(self) -> OllamaHandler:
        if self.tool_handle is not None:
            self.handler.ToolsHandle = self.tool_handle
            self.flow.tools = self.tool_handle
        if not self.handler.model or (self.model and self.handler.model != self.model):
            names = [model.id for model in self.handler.GetModels()]
            target = self.model or (names[0] if names else "")
            if not target:
                raise RuntimeError("Na serwerze nie ma żadnego modelu.")
            if not self.handler.SetModel(target):
                raise RuntimeError(f"Model '{target}' nie jest dostępny na serwerze.")
        return self.handler

    def stream(self, user_text: str) -> Iterator[dict]:
        try:
            self._get_handler()
        except Exception as error:
            Logger.printError(f"Ollama: {error}")
            yield {"type": "error", "message": f"Nie można połączyć z modelem ({self.endpoint}): {error}"}
            return

        snapshot = list(self.flow.context)
        try:
            for event in self.flow.RunStream(user_text, as_events=True):
                if isinstance(event, dict):
                    yield event
        except Exception as error:
            self.flow.context[:] = snapshot
            Logger.printError(f"Ollama: {error}")
            yield {"type": "error", "message": f"Błąd modelu: {error}"}
        finally:
            self.history = self.flow.context


# Interfejsy i zaślepki STT / TTS
class STTBase:

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> str:
        raise NotImplementedError


class TTSBase:

    def speak(self, text: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class StubSTT(STTBase):

    def __init__(self) -> None:
        self.recording = False

    def start(self) -> None:
        self.recording = True

    def stop(self) -> str:
        if not self.recording:
            raise RuntimeError("Nagrywanie nie zostało rozpoczęte.")
        self.recording = False
        return "(tekst z mikrofonu - wersja tymczasowa)"


class StubTTS(TTSBase):

    def speak(self, text: str) -> None:
        Logger.printInfo(f"TTS (zaślepka): {text[:60]}")

    def stop(self) -> None:
        Logger.printInfo("TTS (zaślepka): stop")


class RealSTT(STTBase):
    """Adapter mapujący interfejs STTBase na STTHandle."""

    def __init__(self, handle: STTHandle) -> None:
        self.handle = handle

    def start(self) -> None:
        self.handle.start_recording()

    def stop(self) -> str:
        return self.handle.stop_recording()


class RealTTS(TTSBase):
    """Adapter mapujący interfejs TTSBase na TTSHandle."""

    def __init__(self, handle: TTSHandle) -> None:
        self.handle = handle

    def speak(self, text: str) -> None:
        self.handle.speak_stream((text,))
        if not self.handle.wait():
            raise RuntimeError("Odtwarzanie mowy nie zakończyło się.")
        if self.handle.last_error is not None:
            raise self.handle.last_error

    def stop(self) -> None:
        self.handle.stop()

    def close(self) -> None:
        self.handle.close()


def _missing_audio_dependencies(*modules: str) -> list[str]:
    return [module for module in modules if find_spec(module) is None]


# Funkcje tworzące instancje (miejsce podpięcia AgentLoop, STTHandle, TTSHandle)
def create_agent(settings: dict) -> AgentBase:
    return OllamaAgent(settings["ollama_endpoint"], settings["ollama_model"])


def create_stt(settings: dict | None = None) -> STTBase:
    config = settings or {}
    missing = _missing_audio_dependencies(
        "sounddevice", "numpy", "faster_whisper", "ctranslate2"
    )
    if missing:
        Logger.printWarning(
            f"Moduł STT niedostępny (brak: {', '.join(missing)}) — używam StubSTT."
        )
        return StubSTT()

    handle = STTHandle(
        model_size=config.get("stt_model", "base"),
        device=config.get("stt_device"),
        input_device=config.get("stt_input_device"),
        language=config.get("stt_language", "pl"),
    )
    return RealSTT(handle)


def create_tts(settings: dict | None = None) -> TTSBase:
    config = settings or {}
    missing = _missing_audio_dependencies("sounddevice", "piper", "onnxruntime")
    if missing:
        Logger.printWarning(
            f"Moduł TTS niedostępny (brak: {', '.join(missing)}) — używam StubTTS."
        )
        return StubTTS()

    model_path = config.get("tts_model") or None
    sink = SoundDeviceSink(device=config.get("tts_sink_device"))
    handle = TTSHandle(model_path=model_path, sink=sink)
    return RealTTS(handle)
