"""Narzędzia inspekcji załączników (#51) - dostęp ID-based, bez ścieżek z dysku u modelu.

Model nie "otwiera" plików: narzędzia operują wyłącznie na ID załącznika z bieżącego
czatu, a zakres sesji (chat_id + AttachmentStore) ustawia AgentFlow przed wykonaniem
tool callu. Zwracane metadane (id, nazwa, MIME, wymiary, rozmiar) NIGDY nie zawierają
ścieżki ani katalogu store - to twarde wymaganie #51.

Edycyjne operacje (crop_image, resize_image) i OCR (extract_text_ocr) pozostają
follow-upami zgodnie z checklistem #51.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.Attachments import SEND_MAX_SIDE, AttachmentError  # noqa: F401
from handlers.ToolHandle import ToolHandle, current_tool_session

# list_attachments zwraca najwyżej tyle pozycji - wynik trafia w kontekst modelu.
MAX_LISTED_ATTACHMENTS = 20


def _session_scope() -> Tuple[Optional[object], Optional[str], list]:
    """Zwraca (attachment_store, chat_id, ID załączników z bieżącej wiadomości) albo (None, None, [])."""
    scope = current_tool_session() or {}
    return (
        scope.get("attachment_store"),
        scope.get("chat_id"),
        list(scope.get("attachment_ids") or []),
    )


def _failure(message: str) -> Dict[str, Any]:
    """Błędny wynik narzędzia: komunikat bez ścieżki, katalogu i tracebacku (#51)."""
    return {"success": False, "error": message}


def _record(attachment, index: Optional[int] = None) -> Dict[str, Any]:
    """Metadata-only record of an attachment - deliberately no filesystem path."""
    record: Dict[str, Any] = {
        "id": attachment.id,
        "name": attachment.name or "",
        "mime_type": attachment.mime,
        "width": attachment.width,
        "height": attachment.height,
        "aspect_ratio": (
            round(attachment.width / attachment.height, 3) if attachment.height else None
        ),
        "size_bytes": attachment.size,
        "created_at": attachment.created_at,
    }
    if index is not None:
        record["index"] = index
    return record


def inspect_image(index: int = 1, attachment_id: str = "") -> Dict[str, Any]:
    """Zwraca metadane obrazu z bieżącego czatu - po numeracji w wiadomości albo po ID.

    Args:
        index: 1-based pozycja załącznika (domyślnie pierwszy obraz w bieżącej
            wiadomości; gdy sesja nie zna kolejności - najnowszy obraz czatu).
        attachment_id: pełne 32-hex ID załącznika z tego czatu; ignoruje index,
            gdy podane.

    Returns:
        success + id/name/mime_type/width/height/aspect_ratio/size_bytes/created_at
        oraz normalized_max_side_px (obraz może być zmniejszony przed wysyłką #51).
        Ścieżki na dysku nie są zwracane nigdy.
    """
    store, chat_id, current_ids = _session_scope()
    if store is None or chat_id is None:
        return _failure(
            "Załączniki nie są dostępne w tej sesji agenta (brak przypisanego czatu)."
        )

    target_id: Optional[str] = None
    identifier = (attachment_id or "").strip()
    if identifier:
        target_id = identifier
    else:
        try:
            position = int(index)
        except (TypeError, ValueError):
            return _failure("index musi być liczbą całkowitą (1-based).")
        if current_ids:
            if 1 <= position <= len(current_ids):
                target_id = current_ids[position - 1]
            else:
                return _failure(
                    f"W bieżącej wiadomości jest {len(current_ids)} załączników; "
                    f"index {position} poza zakresem."
                )
        else:
            records = store.list_for_chat(chat_id)
            if not records:
                return _failure("W tym czacie nie ma załączników.")
            if 1 <= position <= len(records):
                target_id = records[position - 1].id
            else:
                return _failure(
                    f"W tym czacie jest {len(records)} załączników; "
                    f"index {position} poza zakresem."
                )

    try:
        attachment = store.get(chat_id, target_id)
    except AttachmentError:
        # Jawnie: nie mówimy modelowi, gdzie plik leży na dysku (#51).
        return _failure("Nie znaleziono załącznika w bieżącym czacie.")

    result = _record(attachment)
    result["normalized_max_side_px"] = SEND_MAX_SIDE
    result["note"] = (
        "Obraz wysyłany do modelu jako data URL; obraz większy niż "
        f"{SEND_MAX_SIDE} px po najdłuższym boku zostaje zmniejszony."
    )
    return {"success": True, **result}


def list_attachments() -> Dict[str, Any]:
    """Listuje załączniki bieżącego czatu (tylko metadane, najnowsze najpierw).

    Zwraca count, attachments[] oraz current_attachment_ids (ID z ostatniej
    wiadomości użytkownika) - wszystko bez ścieżek na dysku (#51).
    """
    store, chat_id, current_ids = _session_scope()
    if store is None or chat_id is None:
        return _failure(
            "Załączniki nie są dostępne w tej sesji agenta (brak przypisanego czatu)."
        )
    records = store.list_for_chat(chat_id)[:MAX_LISTED_ATTACHMENTS]
    return {
        "success": True,
        "count": len(records),
        "attachments": [_record(item, index=i + 1) for i, item in enumerate(records)],
        "current_attachment_ids": current_ids,
    }


IMAGE_TOOL_DEFINITIONS: list = [
    {
        "name": "inspect_image",
        "description": (
            "Inspects one image attachment of the CURRENT chat by its id or 1-based "
            "position. Returns only metadata (id, name, mime type, width, height, "
            "aspect ratio, size in bytes, created time) - never a filesystem path. "
            "Use it to confirm what an attached image is before describing it."
        ),
        "properties": {
            "index": {
                "type": "integer",
                "description": (
                    "1-based position of the attachment in the current message "
                    "(default 1). Ignored when attachment_id is provided."
                ),
            },
            "attachment_id": {
                "type": "string",
                "description": (
                    "32-character hex attachment id from this chat. Optional; omit "
                    "to inspect the first image of the current message."
                ),
            },
        },
        "required": [],
    },
    {
        "name": "list_attachments",
        "description": (
            "Lists image attachments available in the current chat (most recent "
            "first) with metadata only - id, name, mime type, dimensions and size. "
            "No filesystem paths are returned."
        ),
        "properties": {},
        "required": [],
    },
]

AVAILABLE_IMAGE_TOOLS = {
    "inspect_image": inspect_image,
    "list_attachments": list_attachments,
}


def register_tools(tool_handle: ToolHandle) -> None:
    """Rejestruje narzędzia obrazowe przez ToolHandle (#51).

    Zakres (store + chat_id) rozwiązujemy przy każdym wywołaniu przez session
    scope (handlers.ToolHandle), więc jedna rejestracja obsługuje wszystkie czaty
    bez przeciekania stanu między sesjami.
    """
    for definition in IMAGE_TOOL_DEFINITIONS:
        tool_handle.create_tool(**definition)
        name = definition["name"]
        tool_handle.register_function(name, AVAILABLE_IMAGE_TOOLS[name])
