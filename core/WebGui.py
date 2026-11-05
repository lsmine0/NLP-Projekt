import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from core import Logger
from core.AgentAdapter import AgentBase, create_agent, create_stt, create_tts
from core.ChatStore import ChatStore, SettingsStore
from handlers.OllamaHandle import OllamaHandler

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
HOST = "127.0.0.1"
PORT = 8000

app = FastAPI()
store = ChatStore()
settings_store = SettingsStore()
stt = create_stt()
tts = create_tts()
sessions: dict[str, AgentBase] = {}


class ChatRequest(BaseModel):
    chat_id: str | None = None
    message: str


class SpeakRequest(BaseModel):
    text: str


def error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def run_cleanup() -> None:
    cfg = settings_store.get()
    for chat_id in store.cleanup(cfg["max_chats"], cfg["max_age_days"]):
        sessions.pop(chat_id, None)


def get_agent(chat_id: str) -> AgentBase:
    agent = sessions.get(chat_id)
    if agent is None:
        agent = create_agent(settings_store.get())
        agent.load_history(store.get(chat_id)["messages"])
        sessions[chat_id] = agent
    return agent


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


# Endpointy ustawień i listy modeli
@app.get("/api/settings")
def get_settings():
    return settings_store.get()


@app.put("/api/settings")
def put_settings(changes: dict):
    if "ollama_endpoint" in changes and not str(changes["ollama_endpoint"]).strip():
        return error("Adres serwera nie może być pusty.")
    settings = settings_store.update(changes)
    if "ollama_endpoint" in changes or "ollama_model" in changes:
        sessions.clear()
    run_cleanup()
    return settings


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
    return {"ok": True}


# Strumieniowanie odpowiedzi agenta (NDJSON)
@app.post("/api/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        return error("Wiadomość jest pusta.")

    chat_id = request.chat_id
    try:
        if chat_id:
            store.get(chat_id)
        else:
            chat_id = store.create()["id"]
            run_cleanup()
    except KeyError:
        return error("Nie znaleziono czatu.", 404)

    agent = get_agent(chat_id)

    def events():
        yield json.dumps({"type": "chat_id", "id": chat_id}) + "\n"
        store.append(chat_id, "user", message)
        reply: list[str] = []
        thinking: list[str] = []
        completed = False
        try:
            for event in agent.stream(message):
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


if __name__ == "__main__":
    run_cleanup()
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": HOST, "port": PORT},
        daemon=True,
    ).start()
    webview.create_window(
        "Jarvis",
        f"http://{HOST}:{PORT}",
        width=1000,
        height=720,
        min_size=(650, 500),
    )
    webview.start()
