from __future__ import annotations

from typing import Callable
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
import webview
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from core import Logger
from core import Attachments
from core.Attachments import Attachment, AttachmentError, AttachmentStore
from core.AgentAdapter import AgentBase, STTBase, TTSBase, create_agent, create_stt, create_tts
from core.ChatStore import ChatStore, SettingsStore
from handlers.OllamaHandle import OllamaHandler

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
HOST = "127.0.0.1"
PORT = 8000

app = FastAPI()

# Zależności wstrzykiwane przy starcie przez start_gui() (brak inicjalizacji przy imporcie)
store: ChatStore = None  # type: ignore[assignment]
settings_store: SettingsStore = None  # type: ignore[assignment]
stt: STTBase = None  # type: ignore[assignment]
tts: TTSBase = None  # type: ignore[assignment]
attachment_store: AttachmentStore = None  # type: ignore[assignment]
agent_factory: Callable[[dict], AgentBase] = create_agent
audio_enabled: bool = True
sessions: dict[str, AgentBase] = {}


class ChatRequest(BaseModel):
    chat_id: str | None = None
    message: str = ""
    attachments: list[str] = Field(default_factory=list)  # ID z POST /api/attachments


class SpeakRequest(BaseModel):
    text: str


def error(message: str, status: int = 400, code: str | None = None) -> JSONResponse:
    body = {"error": message}
    if code:
        body["code"] = code
    return JSONResponse(body, status_code=status)


# Kody AttachmentError -> status HTTP (GUI rozróżnia błędy po polu "code")
_ATTACHMENT_STATUS = {"empty": 400, "bad_format": 415, "too_large": 413, "not_found": 404}


def attachment_error(exc: AttachmentError) -> JSONResponse:
    return error(str(exc), _ATTACHMENT_STATUS.get(exc.code, 400), exc.code)


def public_attachment(attachment: Attachment) -> dict:
    return {
        key: value for key, value in attachment.to_dict().items()
        if key in ("id", "mime", "size", "width", "height", "name")
    }


def public_settings(settings: dict) -> dict:
    return {
        **settings,
        "audio_enabled": audio_enabled,
        # te same limity co walidacja na backendzie - frontend nie trzyma własnych
        "attachment_limits": {
            "max_bytes": Attachments.MAX_BYTES,
            "max_pixels": Attachments.MAX_PIXELS,
            "max_per_message": Attachments.MAX_PER_MESSAGE,
            "mime_types": [mime for mime, _ in Attachments.ALLOWED_FORMATS.values()],
        },
    }


def run_cleanup() -> None:
    cfg = settings_store.get()
    for chat_id in store.cleanup(cfg["max_chats"], cfg["max_age_days"]):
        sessions.pop(chat_id, None)
        attachment_store.delete_for_chat(chat_id)
    attachment_store.cleanup_pending()


def warmup_model() -> None:
    # Ładuje model do pamięci w tle, żeby pierwsza wiadomość nie płaciła za zimny start.
    try:
        cfg = settings_store.get()
        handler = OllamaHandler(cfg["ollama_endpoint"])
        names = [m.id for m in handler.GetModels()]
        target = cfg["ollama_model"] or (names[0] if names else "")
        if target and handler.SetModel(target):
            handler.Warmup()
    except Exception as exc:
        Logger.printWarning(f"Warmup modelu nieudany: {exc}")


def get_agent(chat_id: str) -> AgentBase:
    agent = sessions.get(chat_id)
    if agent is None:
        agent = agent_factory(settings_store.get())
        agent.load_history(store.get(chat_id)["messages"])
        sessions[chat_id] = agent
    return agent


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


# Endpointy ustawień i listy modeli
@app.get("/api/settings")
def get_settings():
    return public_settings(settings_store.get())


@app.put("/api/settings")
def put_settings(changes: dict):
    if "ollama_endpoint" in changes and not str(changes["ollama_endpoint"]).strip():
        return error("Adres serwera nie może być pusty.")
    if "system_prompt" in changes and not isinstance(changes["system_prompt"], str):
        return error("Prompt systemowy musi być tekstem.")
    settings = settings_store.update(changes)
    if any(key in changes for key in ("ollama_endpoint", "ollama_model", "system_prompt")):
        sessions.clear()
    if "ollama_endpoint" in changes or "ollama_model" in changes:
        threading.Thread(target=warmup_model, daemon=True).start()
    run_cleanup()
    return public_settings(settings)


@app.get("/api/models")
def get_models():
    cfg = settings_store.get()
    try:
        names = [m.id for m in OllamaHandler(cfg["ollama_endpoint"]).GetModels()]
    except Exception as exc:
        Logger.printError(f"Lista modeli: {exc}")
        return error(f"Brak połączenia z serwerem modeli ({cfg['ollama_endpoint']}): {exc}", 502)
    return {"models": names, "current": cfg["ollama_model"] or (names[0] if names else "")}


# Endpointy czatów i wiadomości
@app.get("/api/chats")
def list_chats():
    return store.list()


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    try:
        return store.get(chat_id)
    except KeyError:
        return error("Nie znaleziono czatu.", 404)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    try:
        if not store.delete(chat_id):
            return error("Nie znaleziono czatu.", 404)
    except KeyError:
        return error("Nie znaleziono czatu.", 404)
    sessions.pop(chat_id, None)
    attachment_store.delete_for_chat(chat_id)
    return {"ok": True}


# Załączniki: upload jako oczekujący, przypisanie do czatu dopiero w POST /api/chat
@app.post("/api/attachments")
def upload_attachment(file: UploadFile = File(...)):
    # czytamy o bajt więcej niż limit - wystarczy, żeby wykryć za duży plik
    data = file.file.read(Attachments.MAX_BYTES + 1)
    try:
        attachment = attachment_store.save(data, name=file.filename or "")
    except AttachmentError as exc:
        return attachment_error(exc)
    return public_attachment(attachment)


@app.delete("/api/attachments/{attachment_id}")
def delete_pending_attachment(attachment_id: str):
    try:
        if not attachment_store.delete(None, attachment_id):
            return error("Nie znaleziono załącznika.", 404, "not_found")
    except AttachmentError as exc:
        return attachment_error(exc)
    return {"ok": True}


@app.get("/api/chats/{chat_id}/attachments/{attachment_id}")
def get_attachment(chat_id: str, attachment_id: str):
    try:
        attachment = attachment_store.get(chat_id, attachment_id)
        data = attachment_store.read_bytes(chat_id, attachment_id)
    except AttachmentError as exc:
        return attachment_error(exc)
    return Response(
        content=data,
        media_type=attachment.mime,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=86400",
        },
    )


def check_attachments(chat_id: str | None, attachment_ids: list[str]) -> None:
    # Sprawdza wszystkie ID przed bind(), żeby błąd nie zostawił części przypisanej do czatu
    for attachment_id in attachment_ids:
        if chat_id is not None:
            try:
                attachment_store.get(chat_id, attachment_id)
                continue
            except AttachmentError:
                pass
        attachment_store.get(None, attachment_id)


# Strumieniowanie odpowiedzi agenta (NDJSON)
@app.post("/api/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    attachment_ids = list(dict.fromkeys(request.attachments))
    if not message and not attachment_ids:
        return error("Wiadomość jest pusta.")
    if len(attachment_ids) > Attachments.MAX_PER_MESSAGE:
        return error(
            f"Za dużo załączników (maks. {Attachments.MAX_PER_MESSAGE}).", 400, "too_many"
        )

    chat_id = request.chat_id
    try:
        if chat_id:
            store.get(chat_id)
        check_attachments(chat_id, attachment_ids)
        created = not chat_id
        if created:
            chat_id = store.create()["id"]
        bound = attachment_store.bind(attachment_ids, chat_id)
        # sprzątanie po bind(), żeby cleanup_pending nie usunął sprawdzonych już załączników
        if created:
            run_cleanup()
    except KeyError:
        return error("Nie znaleziono czatu.", 404)
    except AttachmentError as exc:
        return attachment_error(exc)

    agent = get_agent(chat_id)
    # TODO(#51): sprawdzenie vision modelu tutaj, przed zapisem wiadomości do historii

    def events():
        yield json.dumps({"type": "chat_id", "id": chat_id}) + "\n"
        store.append(chat_id, "user", message, attachments=[a.id for a in bound])
        reply: list[str] = []
        thinking: list[str] = []
        completed = False
        # bez załączników wołamy stream(message) - agenci spoza AgentBase nie muszą znać nowego parametru
        stream = agent.stream(message, bound) if bound else agent.stream(message)
        try:
            for event in stream:
                if event["type"] == "thinking":
                    thinking.append(event["text"])
                elif event["type"] == "token":
                    reply.append(event["text"])
                yield json.dumps(event, ensure_ascii=False) + "\n"
            completed = True
            yield json.dumps({"type": "done"}) + "\n"
        finally:
            try:
                if reply or thinking or not completed:
                    text = "".join(reply)
                    if thinking and "<think>" not in text:
                        text = f"<think>{''.join(thinking)}</think>\n{text}"
                    if not completed:
                        text = f"{text}\n<stopped/>".strip()
                    store.append(chat_id, "assistant", text)
            except KeyError:
                pass

    return StreamingResponse(events(), media_type="application/x-ndjson")


# Endpointy głosu (STT / TTS)
@app.post("/api/stt/start")
def stt_start():
    try:
        stt.start()
        return {"ok": True}
    except Exception as exc:
        return error(str(exc), 500)


@app.post("/api/stt/stop")
def stt_stop():
    try:
        return {"text": stt.stop()}
    except Exception as exc:
        return error(str(exc), 500)


@app.post("/api/tts/speak")
def tts_speak(request: SpeakRequest):
    try:
        tts.speak(request.text)
        return {"ok": True}
    except Exception as exc:
        return error(str(exc), 500)


@app.post("/api/tts/stop")
def tts_stop():
    try:
        tts.stop()
        return {"ok": True}
    except Exception as exc:
        return error(str(exc), 500)


def start_gui(
    *,
    chat_store: ChatStore | None = None,
    config_store: SettingsStore | None = None,
    stt_handle: STTBase | None = None,
    tts_handle: TTSBase | None = None,
    attachments: AttachmentStore | None = None,
    custom_agent_factory: Callable[[dict], AgentBase] | None = None,
    audio_enabled: bool = True,
    host: str = HOST,
    port: int = PORT,
) -> None:
    """Funkcja startowa GUI — inicjalizuje przekazane zależności i uruchamia okno."""
    global store, settings_store, stt, tts, agent_factory, attachment_store
    globals()["audio_enabled"] = audio_enabled

    settings_store = config_store if config_store is not None else SettingsStore()
    store = chat_store if chat_store is not None else ChatStore()
    attachment_store = attachments if attachments is not None else AttachmentStore()
    settings = settings_store.get()
    stt = stt_handle if stt_handle is not None else create_stt(settings)
    tts = tts_handle if tts_handle is not None else create_tts(settings)
    if custom_agent_factory is not None:
        agent_factory = custom_agent_factory

    run_cleanup()
    threading.Thread(target=warmup_model, daemon=True).start()
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port},
        daemon=True,
    ).start()
    try:
        webview.create_window(
            "Jarvis",
            f"http://{host}:{port}",
            width=1000,
            height=720,
            min_size=(650, 500),
            text_select=True,
        )
        webview.start()
    finally:
        sessions.clear()


if __name__ == "__main__":
    start_gui()
