from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("CHAT_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
DEFAULT_TITLE = "Nowa rozmowa"
DEFAULT_SYSTEM_PROMPT = (
    "Jesteś uprzejmym, dyskretnym i kompetentnym kamerdynerem. Zwracaj się do "
    "użytkownika z szacunkiem, używając formy „Panie” lub „Pani”, jeśli pasuje "
    "do kontekstu. Mów spokojnie, elegancko i rzeczowo. Gdy otrzymasz prośbę, "
    "potwierdź ją krótko w stylu „Oczywiście, już się tym zajmuję” i pomóż "
    "najlepiej, jak potrafisz. Nie twierdź, że wykonałeś czynność poza czatem, "
    "jeśli faktycznie jej nie wykonałeś; jasno informuj o ograniczeniach i "
    "dopytuj, gdy brakuje ważnych informacji."
)
_MERMAID_BLOCK_RE = re.compile(
    r"(?im)^ {0,3}(?:`{3,}|~{3,})[ \t]*mermaid(?:[ \t]+[^\r\n]*)?\r?\n"
    r"(?P<source>[\s\S]*?)^ {0,3}(?:`{3,}|~{3,})[ \t]*$"
)


def _normalize_mermaid_source(source: str) -> str:
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.strip() for line in lines)


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

    # attachments: lista ID z core/Attachments.py (w JSON czatu nie trzymamy danych obrazu)
    def append(
        self,
        chat_id: str,
        role: str,
        content: str,
        attachments: list[str] | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        chat = self.get(chat_id)
        message = {"role": role, "content": content}
        if attachments:
            message["attachments"] = list(attachments)
        if tools:
            # Presentation metadata only (name/source/risk/status), never arguments.
            message["tools"] = list(tools)
        chat["messages"].append(message)
        chat["updated_at"] = _now()
        if role == "user" and chat["title"] == DEFAULT_TITLE:
            chat["title"] = content.strip()[:28] or DEFAULT_TITLE
        _write_json(self._path(chat_id), chat)
        return chat

    def replace_last_mermaid_source(
        self, chat_id: str, original_source: str, corrected_source: str
    ) -> bool:
        chat = self.get(chat_id)
        expected = _normalize_mermaid_source(original_source)
        for message in reversed(chat["messages"]):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            matches = list(_MERMAID_BLOCK_RE.finditer(content))
            for match in reversed(matches):
                if _normalize_mermaid_source(match.group("source")) != expected:
                    continue
                replacement = corrected_source.strip("\r\n") + "\n"
                message["content"] = (
                    content[:match.start("source")]
                    + replacement
                    + content[match.end("source"):]
                )
                chat["updated_at"] = _now()
                _write_json(self._path(chat_id), chat)
                return True
        return False

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
            "ollama_endpoint": os.environ.get("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/v1"),
            "ollama_model": os.environ.get("OLLAMA_MODEL", ""),
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "max_chats": 50,
            "max_age_days": 30,
            "tts_enabled": False,
            "stt_auto_send": False,
            "stt_model": "base",
            "stt_device": None,
            "stt_input_device": None,
            "stt_language": "pl",
            "tts_model": "",
            "tts_sink_device": None,
            "theme": "dark",
            "show_history": True,
            "tool_policies": {},
            "tool_risk_policies": {},
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
