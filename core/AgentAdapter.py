from typing import Iterator
from core import Logger
from handlers.ContextHandle import ContextHandle
from handlers.OllamaHandle import OllamaHandler


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


# Czat z modelem przez OllamaHandler
class OllamaAgent(AgentBase):

    def __init__(self, endpoint: str, model: str = "", system_prompt: str | None = None) -> None:
        self.endpoint = endpoint
        self.model = model
        self.system_prompt = system_prompt
        self.handler: OllamaHandler | None = None
        self.history: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.history = []
        if self.system_prompt:
            self.history.append(ContextHandle.Create(self.system_prompt))

    def load_history(self, messages: list[dict]) -> None:
        self.reset()
        self.history.extend(
            ContextHandle.Create(message["content"], message["role"])
            for message in messages
            if message.get("role") in ("user", "assistant")
        )

    def _get_handler(self) -> OllamaHandler:
        if self.handler is None:
            self.handler = OllamaHandler(self.endpoint)
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
            handler = self._get_handler()
        except Exception as error:
            Logger.printError(f"Ollama: {error}")
            yield {"type": "error", "message": f"Nie można połączyć z modelem ({self.endpoint}): {error}"}
            return

        self.history.append(ContextHandle.Create(user_text, "user"))
        reply: list[str] = []
        try:
            for chunk in handler.ChatStream(self.history):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if thinking:
                    yield {"type": "thinking", "text": thinking}
                text = delta.content
                if text:
                    reply.append(text)
                    yield {"type": "token", "text": text}
        except Exception as error:
            Logger.printError(f"Ollama: {error}")
            yield {"type": "error", "message": f"Błąd modelu: {error}"}
        finally:
            if reply:
                self.history.append(ContextHandle.Create("".join(reply), "assistant"))
            else:
                self.history.pop()


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


# Funkcje tworzące instancje (miejsce podpięcia AgentLoop, STTHandle, TTSHandle)
def create_agent(settings: dict) -> AgentBase:
    return OllamaAgent(settings["ollama_endpoint"], settings["ollama_model"])


def create_stt() -> STTBase:
    return StubSTT()


def create_tts() -> TTSBase:
    return StubTTS()
