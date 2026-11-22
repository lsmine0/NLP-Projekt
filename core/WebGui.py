from __future__ import annotations

from typing import Callable
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import APIRouter, FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from core import Logger
from core import Attachments
from core.Attachments import Attachment, AttachmentError, AttachmentStore
from core.AgentAdapter import AgentBase, STTBase, TTSBase, create_agent, create_stt, create_tts
from core.ChatStore import ChatStore, SettingsStore
from core.RuntimeConfig import RuntimeConfig
from dataclasses import dataclass, field
from types import SimpleNamespace
from contextlib import asynccontextmanager

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
HOST = "127.0.0.1"
PORT = 8000

router = APIRouter()

# Zgodność z dotychczasowymi bezpośrednimi wywołaniami/testami.
# Nowe instancje ASGI używają wyłącznie własnego app.state.api w lifespan.
store: ChatStore = None  # type: ignore[assignment]
settings_store: SettingsStore = None  # type: ignore[assignment]
stt: STTBase = None  # type: ignore[assignment]
tts: TTSBase = None  # type: ignore[assignment]
attachment_store: AttachmentStore = None  # type: ignore[assignment]
agent_factory: Callable[[dict], AgentBase] = create_agent
audio_enabled: bool = True
sessions: dict[str, AgentBase] = {}


@dataclass
class ApiState:
    store: ChatStore
    settings_store: SettingsStore
    stt: STTBase
    tts: TTSBase
    attachment_store: AttachmentStore
    agent_factory: Callable[[dict], AgentBase]
    audio_enabled: bool
    runtime: object
    sessions: dict = field(default_factory=dict)


def _state(request=None):
    if request is not None and hasattr(request.app.state, 'api'):
        return request.app.state.api
    return SimpleNamespace(store=store, settings_store=settings_store, stt=stt,
                           tts=tts, attachment_store=attachment_store,
                           agent_factory=agent_factory, audio_enabled=audio_enabled,
                           sessions=sessions, runtime=None)



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
_ATTACHMENT_STATUS = {
    "empty": 400,
    "bad_format": 415,
    "too_large": 413,
    "not_found": 404,
    # #51: model bez capability 'vision' - obraz nie wysłany, message nie zapisana.
    "vision_unsupported": 415,
}


def attachment_error(exc: AttachmentError) -> JSONResponse:
    return error(str(exc), _ATTACHMENT_STATUS.get(exc.code, 400), exc.code)


def agent_vision_support(agent) -> bool | None:
    """Próbujemy dowiedzieć się, czy model agenta akceptuje obrazki (#51).

    Zwraca True/False albo None, gdy probe nie jest możliwa (atrapa w testach,
    provider bez metadanych) – wtedy GUI nie blokuje wysyłki. False = jawnie
    unsupported, więc obraz nie poleci jako plain text.
    """
    probe = getattr(agent, "supports_vision", None)
    if not callable(probe):
        return None
    try:
        result = probe()
    except Exception:
        Logger.printWarning("Nie można ustalić capability 'vision' modelu.")
        return None
    return result if isinstance(result, bool) else None


def public_attachment(attachment: Attachment) -> dict:
    return {
        key: value for key, value in attachment.to_dict().items()
        if key in ("id", "mime", "size", "width", "height", "name")
    }


def public_settings(settings: dict, request: Request = None) -> dict:
    state = _state(request)
    audio_enabled = state.audio_enabled
    if state.runtime is not None and getattr(state.runtime, 'config_locked', False):
        settings = dict(settings, ollama_endpoint=state.runtime.config.endpoint,
                        ollama_model=state.runtime.config.model or settings.get('ollama_model', ''))
    return {
        **{k: v for k, v in settings.items() if k in SettingsStore.defaults()},
        "audio_enabled": audio_enabled,
        # te same limity co walidacja na backendzie - frontend nie trzyma własnych
        "attachment_limits": {
            "max_bytes": Attachments.MAX_BYTES,
            "max_pixels": Attachments.MAX_PIXELS,
            "max_per_message": Attachments.MAX_PER_MESSAGE,
            "mime_types": [mime for mime, _ in Attachments.ALLOWED_FORMATS.values()],
        },
    }


def run_cleanup(request: Request = None) -> None:
    state = _state(request)
    store = state.store
    settings_store = state.settings_store
    attachment_store = state.attachment_store
    sessions = state.sessions
    cfg = settings_store.get()
    for chat_id in store.cleanup(cfg["max_chats"], cfg["max_age_days"]):
        sessions.pop(chat_id, None)
        attachment_store.delete_for_chat(chat_id)
    attachment_store.cleanup_pending()


def warmup_model(request: Request = None) -> None:
    state = _state(request)
    if state.runtime is None:
        return
    try:
        handler = state.runtime.provider
        names = [m.id for m in handler.GetModels()]
        target = (state.runtime.config.model if state.runtime.config_locked else '') or state.settings_store.get()['ollama_model'] or (names[0] if names else '')
        if target and handler.SetModel(target):
            warmup = getattr(handler, 'Warmup', None)
            if callable(warmup):
                warmup()
    except Exception:
        Logger.printWarning('Warmup modelu nieudany.')


def get_agent(chat_id: str, request: Request = None) -> AgentBase:
    state = _state(request)
    store = state.store
    settings_store = state.settings_store
    sessions = state.sessions
    agent_factory = state.agent_factory
    attachment_store = state.attachment_store
    agent = sessions.get(chat_id)
    if agent is None:
        agent = agent_factory(settings_store.get())
        # #51: spina store z sesją agenta (per czat) - ID -> data URL przy wysyłce,
        # nigdy ścieżka z dysku. Agenci-atrapy bez bindera pracują jak dotychczas.
        binder = getattr(agent, "bind_attachments", None)
        if callable(binder):
            try:
                binder(attachment_store, chat_id)
            except Exception as exc:
                Logger.printWarning(
                    f"Nie można spiąć magazynu załączników z sesją: {exc}"
                )
        agent.load_history(store.get(chat_id)["messages"])
        sessions[chat_id] = agent
    return agent


@router.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


# Endpointy ustawień i listy modeli
@router.get("/api/settings")
def get_settings(request: Request = None):
    state = _state(request)
    settings_store = state.settings_store
    return public_settings(settings_store.get(), request)


@router.put("/api/settings")
def put_settings(changes: dict, request: Request = None):
    state = _state(request)
    settings_store = state.settings_store
    sessions = state.sessions
    if state.runtime is not None and getattr(state.runtime, 'config_locked', False):
        for key, value in (('ollama_endpoint', state.runtime.config.endpoint),
                           ('ollama_model', state.runtime.config.model or settings_store.get().get('ollama_model', ''))):
            if key in changes and changes[key] != value:
                return error('Konfiguracja modelu API jest ustawiana przy starcie procesu (NLP_ENDPOINT/NLP_MODEL).')
    if any(k.lower() in ('api_key', 'openai_api_key', 'ollama_api_key', 'authorization', 'token') for k in changes):
        return error('Klucze API należy ustawić w środowisku procesu, nie w settings.json.')
    if 'ollama_endpoint' in changes:
        try:
            RuntimeConfig(endpoint=str(changes['ollama_endpoint']))
        except ValueError:
            return error('Nieprawidłowy endpoint (wymagany HTTP(S), bez danych logowania).')
    if "ollama_endpoint" in changes and not str(changes["ollama_endpoint"]).strip():
        return error("Adres serwera nie może być pusty.")
    if "system_prompt" in changes and not isinstance(changes["system_prompt"], str):
        return error("Prompt systemowy musi być tekstem.")
    settings = settings_store.update(changes)
    if any(key in changes for key in ("ollama_endpoint", "ollama_model", "system_prompt")):
        sessions.clear()
    if "ollama_endpoint" in changes or "ollama_model" in changes:
        if state.runtime is not None and not state.runtime.config_locked:
            state.runtime.provider = state.runtime.provider_factory(settings)
        threading.Thread(target=warmup_model, args=(request,), daemon=True).start()
    run_cleanup(request)
    return public_settings(settings, request)


@router.get("/api/models")
def get_models(request: Request = None):
    state = _state(request)
    settings_store = state.settings_store
    try:
        handler = state.runtime.provider
        names = [m.id for m in handler.GetModels()]
    except Exception:
        return error('Provider modelu jest niedostępny.', 502)
    return {'models': names, 'current': (state.runtime.config.model if state.runtime.config_locked else '') or settings_store.get()['ollama_model'] or (names[0] if names else '')}


# Endpointy czatów i wiadomości
@router.get("/api/chats")
def list_chats(request: Request = None):
    state = _state(request)
    store = state.store
    return store.list()


@router.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, request: Request = None):
    state = _state(request)
    store = state.store
    try:
        return store.get(chat_id)
    except KeyError:
        return error("Nie znaleziono czatu.", 404)


@router.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, request: Request = None):
    state = _state(request)
    store = state.store
    attachment_store = state.attachment_store
    sessions = state.sessions
    try:
        if not store.delete(chat_id):
            return error("Nie znaleziono czatu.", 404)
    except KeyError:
        return error("Nie znaleziono czatu.", 404)
    sessions.pop(chat_id, None)
    attachment_store.delete_for_chat(chat_id)
    return {"ok": True}


# Załączniki: upload jako oczekujący, przypisanie do czatu dopiero w POST /api/chat
@router.post("/api/attachments")
def upload_attachment(file: UploadFile = File(...), request: Request = None):
    state = _state(request)
    attachment_store = state.attachment_store
    # czytamy o bajt więcej niż limit - wystarczy, żeby wykryć za duży plik
    data = file.file.read(Attachments.MAX_BYTES + 1)
    try:
        attachment = attachment_store.save(data, name=file.filename or "")
    except AttachmentError as exc:
        return attachment_error(exc)
    return public_attachment(attachment)


@router.delete("/api/attachments/{attachment_id}")
def delete_pending_attachment(attachment_id: str, request: Request = None):
    state = _state(request)
    attachment_store = state.attachment_store
    try:
        if not attachment_store.delete(None, attachment_id):
            return error("Nie znaleziono załącznika.", 404, "not_found")
    except AttachmentError as exc:
        return attachment_error(exc)
    return {"ok": True}


@router.get("/api/chats/{chat_id}/attachments/{attachment_id}")
def get_attachment(chat_id: str, attachment_id: str, request: Request = None):
    state = _state(request)
    attachment_store = state.attachment_store
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


def check_attachments(chat_id: str | None, attachment_ids: list[str], request: Request = None) -> None:
    state = _state(request)
    attachment_store = state.attachment_store
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
@router.post("/api/chat")
def chat(request: ChatRequest, http_request: Request = None):
    state = _state(http_request)
    store = state.store
    attachment_store = state.attachment_store
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
        check_attachments(chat_id, attachment_ids, http_request)
        created = not chat_id
        if created:
            chat_id = store.create()["id"]
        bound = attachment_store.bind(attachment_ids, chat_id)
        # sprzątanie po bind(), żeby cleanup_pending nie usunął sprawdzonych już załączników
        if created:
            run_cleanup(http_request)
    except KeyError:
        return error("Nie znaleziono czatu.", 404)
    except AttachmentError as exc:
        return attachment_error(exc)

    agent = get_agent(chat_id, http_request)
    # #51: pre-flight capability 'vision' PRZED zapisaniem message w history - model
    # bez vision nie może dostać obrazu jako plain text (defence-in-depth: identyczny
    # gate jest tez w AgentAdapter.OllamaAgent.stream()).
    if attachment_ids and agent_vision_support(agent) is False:
        Logger.printWarning("Model bez capability 'vision' - wysyłka obrazu zablokowana.")
        return error(
            "Wybrany model nie obsługuje analizy obrazów (capability 'vision'). "
            "Zmień model w ustawieniach albo usuń załączniki.",
            415,
            "vision_unsupported",
        )

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
@router.post("/api/stt/start")
def stt_start(request: Request = None):
    state = _state(request)
    stt = state.stt
    if not state.audio_enabled:
        return error('Obsługa audio jest wyłączona.', 503)
    try:
        stt.start()
        return {"ok": True}
    except Exception as exc:
        return error(str(exc), 500)


@router.post("/api/stt/stop")
def stt_stop(request: Request = None):
    state = _state(request)
    stt = state.stt
    if not state.audio_enabled:
        return error('Obsługa audio jest wyłączona.', 503)
    try:
        return {"text": stt.stop()}
    except Exception as exc:
        return error(str(exc), 500)


@router.post("/api/tts/speak")
def tts_speak(request: SpeakRequest, http_request: Request = None):
    state = _state(http_request)
    tts = state.tts
    if not state.audio_enabled:
        return error('Obsługa audio jest wyłączona.', 503)
    try:
        tts.speak(request.text)
        return {"ok": True}
    except Exception as exc:
        return error(str(exc), 500)


@router.post("/api/tts/stop")
def tts_stop(request: Request = None):
    state = _state(request)
    tts = state.tts
    if not state.audio_enabled:
        return error('Obsługa audio jest wyłączona.', 503)
    try:
        tts.stop()
        return {"ok": True}
    except Exception as exc:
        return error(str(exc), 500)



@router.get('/health/live')
def liveness():
    return {'status': 'alive'}


@router.get('/health/ready')
def readiness(request: Request):
    state = getattr(request.app.state, 'api', None)
    components = {'runtime': state is not None, 'provider': False, 'agent': False}
    if state is not None:
        components['agent'] = callable(state.agent_factory)
        try:
            provider = state.runtime.provider
            names = [m.id for m in provider.GetModels()]
            model = (state.runtime.config.model if state.runtime.config_locked else '') or state.settings_store.get().get('ollama_model')
            components['provider'] = bool(names) and (not model or model in names)
        except Exception:
            pass  # Never expose endpoint, credentials, paths or raw exception text.
    ready = all(components.values())
    return JSONResponse({'status': 'ready' if ready else 'not_ready',
                         'components': components}, status_code=200 if ready else 503)


def create_app(*, runtime=None, runtime_factory=None):
    """ASGI factory: dependencies are created in lifespan, never on module import."""
    @asynccontextmanager
    async def lifespan(application):
        owned = runtime is None
        if runtime is not None:
            current = runtime
        elif runtime_factory is not None:
            current = runtime_factory()
        else:
            from core.main import build_api_runtime
            current = build_api_runtime()
        try:
            state = ApiState(
                store=current.chat_store, settings_store=current.settings_store,
                stt=current.stt, tts=current.tts,
                # #51: preferuj store z runtime (NLP_WORKSPACE/CHAT_DATA_DIR); fallback
                # dla starych AppRuntime bez tego pola.
                attachment_store=(getattr(current, 'attachment_store', None)
                                  or AttachmentStore(current.settings_store.path.parent / 'attachments')),
                agent_factory=current.agent_factory, audio_enabled=current.audio_enabled,
                runtime=current,
            )
            application.state.api = state
            # Cleanup uses this app's state, not the compatibility globals.
            run_cleanup(SimpleNamespace(app=application))
            yield
        finally:
            if hasattr(application.state, 'api'):
                application.state.api.sessions.clear()
                del application.state.api
            if owned:
                current.close()

    application = FastAPI(lifespan=lifespan)
    application.include_router(router)
    return application

def start_gui(
    *,
    runtime=None,
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
    """Desktop launcher; WebView is optional and imported only here."""
    import webview
    if runtime is None:
        from core.main import build_runtime, parse_args
        args, _ = parse_args(['--no-audio'] if not audio_enabled else [])
        runtime = build_runtime(args)
        owned = True
        if chat_store is not None:
            runtime.chat_store = chat_store
        if config_store is not None:
            runtime.settings_store = config_store
        if stt_handle is not None:
            runtime.stt = stt_handle
        if tts_handle is not None:
            runtime.tts = tts_handle
        if custom_agent_factory is not None:
            runtime.agent_factory = custom_agent_factory
    else:
        owned = False
    application = create_app(runtime=runtime)
    server = uvicorn.Server(uvicorn.Config(application, host=host, port=port))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        webview.create_window('Jarvis', f'http://{host}:{port}', width=1000,
                              height=720, min_size=(650, 500), text_select=True)
        webview.start()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if owned:
            runtime.close()


app = create_app()

if __name__ == '__main__':
    start_gui()
