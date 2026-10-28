import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("CHAT_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
DEFAULT_TITLE = "Nowa rozmowa"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Zapis czatów w plikach data/chats/<id>.json
class ChatStore:

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else DATA_DIR / "chats"

    def _path(self, chat_id: str) -> Path:
        safe_id = Path(chat_id or "").name
        if not safe_id or safe_id != chat_id:
            raise KeyError(chat_id)
        return self.directory / f"{safe_id}.json"

    def create(self, title: str = DEFAULT_TITLE) -> dict:
        chat = {
            "id": uuid.uuid4().hex,
            "title": title,
            "updated_at": _now(),
            "messages": [],
        }
        _write_json(self._path(chat["id"]), chat)
        return chat

    def get(self, chat_id: str) -> dict:
        path = self._path(chat_id)
        if not path.is_file():
            raise KeyError(chat_id)
        return json.loads(path.read_text(encoding="utf-8"))

    # Zwraca listę czatów (bez wiadomości) od najnowszego
    def list(self) -> list[dict]:
        summaries = []
        if self.directory.is_dir():
            for path in self.directory.glob("*.json"):
                try:
                    chat = json.loads(path.read_text(encoding="utf-8"))
                    summaries.append({
                        "id": chat["id"],
                        "title": chat["title"],
                        "updated_at": chat["updated_at"],
                    })
                except (OSError, ValueError, KeyError):
                    continue
        summaries.sort(key=lambda c: c["updated_at"], reverse=True)
        return summaries

    def append(self, chat_id: str, role: str, content: str) -> dict:
        chat = self.get(chat_id)
        chat["messages"].append({"role": role, "content": content})
        chat["updated_at"] = _now()
        if role == "user" and chat["title"] == DEFAULT_TITLE:
            chat["title"] = content.strip()[:28] or DEFAULT_TITLE
        _write_json(self._path(chat_id), chat)
        return chat

    def delete(self, chat_id: str) -> bool:
        path = self._path(chat_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # Usuwa stare czaty (0 wyłącza limit)
    def cleanup(self, max_chats: int = 0, max_age_days: int = 0) -> list[str]:
        removed = []
        chats = self.list()

        if max_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            for chat in list(chats):
                if datetime.fromisoformat(chat["updated_at"]) < cutoff:
                    self.delete(chat["id"])
                    removed.append(chat["id"])
                    chats.remove(chat)

        if max_chats > 0:
            for chat in chats[max_chats:]:
                self.delete(chat["id"])
                removed.append(chat["id"])
        return removed


# Zapis ustawień GUI w pliku data/settings.json
class SettingsStore:

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DATA_DIR / "settings.json"

    @staticmethod
    def defaults() -> dict:
        return {
            "ollama_endpoint": os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
            "ollama_model": os.environ.get("OLLAMA_MODEL", ""),
            "max_chats": 50,
            "max_age_days": 30,
            "tts_enabled": False,
            "stt_auto_send": False,
            "theme": "dark",
            "show_history": True,
        }

    def get(self) -> dict:
        settings = self.defaults()
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                settings.update({k: v for k, v in saved.items() if k in settings})
            except (OSError, ValueError):
                pass
        return settings

    def update(self, changes: dict) -> dict:
        settings = self.get()
        settings.update({k: v for k, v in changes.items() if k in settings and v is not None})
        _write_json(self.path, settings)
        return settings
