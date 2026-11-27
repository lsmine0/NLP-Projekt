from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
EXPORT_FORMATS: dict[str, str] = {
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
}
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think\s*>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>[\s\S]*\Z", re.IGNORECASE)
_THINK_CLOSE_STRAY_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_STOPPED_TAG_RE = re.compile(r"<stopped\s*/?>", re.IGNORECASE)
_ROLE_LABELS: dict[str, str] = {
    "user": "Użytkownik",
    "assistant": "Asystent",
}
_INTERRUPTED_TEXT = "Odpowiedź została przerwana przez użytkownika"


@dataclass(frozen=True)
class ExportedChat:
    content: str
    filename: str
    media_type: str
    format: str


def normalize_export_format(fmt: str | None) -> str:
    normalized = (fmt or "").strip().lower()
    if normalized not in EXPORT_FORMATS:
        raise ValueError("Nieobsługiwany format eksportu. Dostępne formaty: txt, md.")
    return normalized


def safe_export_filename(chat: dict, fmt: str) -> str:
    normalized_fmt = normalize_export_format(fmt)
    raw_title = str(chat.get("title") or "rozmowa").replace("ł", "l").replace("Ł", "L")
    ascii_title = (
        unicodedata.normalize("NFKD", raw_title)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9_-]+", "-", ascii_title).strip("-_")[:48] or "rozmowa"
    raw_id = re.sub(r"[^a-zA-Z0-9]+", "", str(chat.get("id") or ""))
    short_id = raw_id[:8]
    stem = f"{slug}-{short_id}" if short_id else slug
    return f"{stem}.{normalized_fmt}"


def _sanitize_assistant_content(raw_content: str) -> tuple[str, bool]:
    text = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    stopped = bool(_STOPPED_TAG_RE.search(text))
    text = _STOPPED_TAG_RE.sub("", text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    text = _THINK_CLOSE_STRAY_RE.sub("", text)
    return text.strip(), stopped


def _resolve_export_attachments(
    chat_id: str, attachment_ids: list, attachment_store: Any = None
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for raw_id in attachment_ids:
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        att_id = raw_id.strip()
        name = ""
        mime = ""
        if attachment_store is not None:
            try:
                attachment = attachment_store.get(chat_id, att_id)
                name = (
                    str(getattr(attachment, "name", "") or "")
                    .replace("\\", "/")
                    .rsplit("/", 1)[-1]
                    .strip()
                )
                mime = str(getattr(attachment, "mime", "") or "").strip()
            except Exception:
                pass
        items.append({"id": att_id, "name": name, "mime": mime})
    return items


def _format_attachment_line(item: dict[str, str], *, markdown: bool) -> str:
    att_id = f"`{item['id']}`" if markdown else item["id"]
    name = item.get("name", "")
    mime = item.get("mime", "")
    if name and mime:
        details = f"{name} ({mime}, ID: {att_id})"
    elif name:
        details = f"{name} (ID: {att_id})"
    elif mime:
        details = f"{mime} (ID: {att_id})"
    else:
        details = f"ID: {att_id}"
    prefix = "- Załącznik: " if markdown else "Załącznik: "
    return f"{prefix}{details}"


def format_chat_export(
    chat: dict, fmt: str, *, attachment_store: Any = None
) -> ExportedChat:
    normalized_fmt = normalize_export_format(fmt)
    is_md = normalized_fmt == "md"
    chat_id = str(chat.get("id") or "")
    title = str(chat.get("title") or DEFAULT_TITLE).strip() or DEFAULT_TITLE
    updated_at = str(chat.get("updated_at") or "").strip()
    raw_messages = chat.get("messages") or []

    lines: list[str] = []
    if is_md:
        lines.append(f"# {title}")
        lines.append("")
        if chat_id:
            lines.append(f"- ID rozmowy: `{chat_id}`")
        if updated_at:
            lines.append(f"- Ostatnia aktualizacja: {updated_at}")
        lines.extend(["", "---", ""])
    else:
        lines.append(title)
        if chat_id:
            lines.append(f"ID rozmowy: {chat_id}")
        if updated_at:
            lines.append(f"Ostatnia aktualizacja: {updated_at}")
        lines.extend(["=" * 60, ""])

    if not raw_messages:
        lines.append("*Brak wiadomości w rozmowie.*" if is_md else "Brak wiadomości w rozmowie.")
    else:
        rendered_blocks: list[str] = []
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            role_label = _ROLE_LABELS.get(role, role.capitalize() if role else "Wiadomość")
            raw_content = str(message.get("content") or "")

            att_items = _resolve_export_attachments(
                chat_id,
                list(message.get("attachments") or []),
                attachment_store=attachment_store,
            )
            att_lines = [_format_attachment_line(item, markdown=is_md) for item in att_items]

            body_parts: list[str] = []
            if att_lines:
                body_parts.append("\n".join(att_lines))

            if role == "assistant":
                cleaned, stopped = _sanitize_assistant_content(raw_content)
                if cleaned:
                    body_parts.append(cleaned)
                if stopped:
                    notice = f"*[{_INTERRUPTED_TEXT}]*" if is_md else f"[{_INTERRUPTED_TEXT}]"
                    body_parts.append(notice)
                elif not cleaned and not att_lines:
                    body_parts.append("*(brak treści odpowiedzi)*" if is_md else "(brak treści odpowiedzi)")
            else:
                cleaned_user = raw_content.replace("\r\n", "\n").replace("\r", "\n").strip()
                if cleaned_user:
                    body_parts.append(cleaned_user)
                elif not att_lines:
                    body_parts.append("*(pusta wiadomość)*" if is_md else "(pusta wiadomość)")

            header = f"## {role_label}" if is_md else f"[{role_label}]"
            rendered_blocks.append(f"{header}\n\n" + "\n\n".join(body_parts))

        if not rendered_blocks:
            lines.append("*Brak wiadomości w rozmowie.*" if is_md else "Brak wiadomości w rozmowie.")
        else:
            separator = "\n\n" if is_md else f"\n\n{'-' * 60}\n\n"
            lines.append(separator.join(rendered_blocks))

    rendered = "\n".join(lines).rstrip() + "\n"
    return ExportedChat(
        content=rendered,
        filename=safe_export_filename(chat, normalized_fmt),
        media_type=EXPORT_FORMATS[normalized_fmt],
        format=normalized_fmt,
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

    def export_chat(
        self,
        chat_id: str,
        fmt: str,
        *,
        attachment_store: Any = None,
    ) -> ExportedChat:
        normalized_fmt = normalize_export_format(fmt)
        chat = self.get(chat_id)
        return format_chat_export(chat, normalized_fmt, attachment_store=attachment_store)

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
