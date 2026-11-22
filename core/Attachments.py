"""Wspólny format załączników (#51 / #52): rekord, walidacja obrazu i magazyn plików.

Układ na dysku (DATA_DIR/attachments):
    _pending/<id>.<ext> + <id>.json   - wysłane przez GUI, jeszcze nieprzypisane do czatu
    <chat_id>/<id>.<ext> + <id>.json  - przypisane do wiadomości w czacie

Na zewnątrz (GUI, model, narzędzia) krąży wyłącznie ID załącznika - nigdy ścieżka.
Dostęp zawsze przez parę (chat_id, id), więc załącznika z innego czatu nie da się odczytać.
"""
from __future__ import annotations

import base64
import io
import json
import re
import shutil
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from core.ChatStore import DATA_DIR

# Jedno źródło limitów - używane przy uploadzie (#52) i przed wysłaniem do modelu (#51)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_PER_MESSAGE = 4
ALLOWED_FORMATS: dict[str, tuple[str, str]] = {
    # format Pillow -> (MIME, rozszerzenie pliku); SVG celowo pominięte (może zawierać skrypty)
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "WEBP": ("image/webp", "webp"),
    "GIF": ("image/gif", "gif"),
}
PENDING_DIR = "_pending"
PENDING_TTL_HOURS = 24

# Limity wysyłki do modelu (#51) - normalizacja zachodzi dopiero przy budowaniu
# zapytania, więc oryginał w store zostaje nietknięty dla GUI (#51).
SEND_MAX_SIDE = 2048                        # najdłuższy bok po normalizacji
SEND_MAX_IMAGE_BYTES = 2 * 1024 * 1024      # max jeden obraz po re-encode
SEND_MIN_QUALITY = 55                       # najniżej przy step-down JPEG
SEND_TOTAL_BUDGET = 6 * 1024 * 1024         # max suma data URL w jednym zapytaniu
RESOLVER_CACHE_SIZE = 32                    # prepared data URL per attachment id

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class AttachmentError(ValueError):
    """Błąd załącznika z kodem do obsługi w GUI: empty, too_large, bad_format, not_found."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Attachment:
    id: str
    mime: str
    size: int
    width: int
    height: int
    chat_id: str | None
    created_at: str
    name: str = ""  # oryginalna nazwa pliku - tylko do wyświetlenia

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Attachment":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    @property
    def extension(self) -> str:
        return next(ext for mime, ext in ALLOWED_FORMATS.values() if mime == self.mime)


def validate_image(data: bytes) -> tuple[str, int, int]:
    """Sprawdza obraz po zawartości (nie po rozszerzeniu). Zwraca (mime, width, height)."""
    if not data:
        raise AttachmentError("empty", "Plik jest pusty.")
    if len(data) > MAX_BYTES:
        raise AttachmentError(
            "too_large", f"Plik jest za duży (maks. {MAX_BYTES // (1024 * 1024)} MB)."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in ALLOWED_FORMATS:
                raise AttachmentError(
                    "bad_format", "Nieobsługiwany format. Dozwolone: PNG, JPEG, WEBP, GIF."
                )
            if width * height > MAX_PIXELS:
                raise AttachmentError("too_large", "Obraz ma za dużą rozdzielczość.")
            image.verify()
    except AttachmentError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise AttachmentError("bad_format", "Plik nie jest poprawnym obrazem.") from None
    return ALLOWED_FORMATS[image_format][0], width, height


def _encode_image(image, image_format: str, quality: int | None = None) -> bytes:
    """Koduje obraz do bytes; JPEG wymusza RGB (#51)."""
    buffer = io.BytesIO()
    if image_format == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=quality or 85)
    else:
        image.save(buffer, format=image_format)
    return buffer.getvalue()


def prepare_image(data: bytes, *, max_side: int = SEND_MAX_SIDE) -> tuple[bytes, str, int, int]:
    """Zmniejsza/przekodowuje obraz przed wysłaniem do modelu (#51).

    Zwraca (data, mime, width, height). Najpierw walidacja po zawartości
    (validate_image = jedno źródło limitów), potem resize tylko gdy obraz
    przekracza max_side - mniejsze pliki lecą bez re-encode (zero strat jakości/CPU).
    Animowany GIF -> pierwsza klatka (model i tak zobaczyłby tylko ją), kodowana do
    PNG, żeby płaskie kolory nie zarobi artefaktów JPEG.
    """
    mime, width, height = validate_image(data)
    if max_side and max(width, height) <= max_side:
        return data, mime, width, height

    scale = max_side / float(max(width, height))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        with Image.open(io.BytesIO(data)) as image:
            source_format = image.format
            if source_format == "GIF" and getattr(image, "n_frames", 1) > 1:
                image.seek(0)
            keep_alpha = source_format == "PNG" and "A" in (image.mode or "")
            working = image if keep_alpha else image.convert("RGB")
            resized = working.resize(new_size, Image.LANCZOS)
    except (OSError, ValueError, SyntaxError):
        raise AttachmentError("bad_format", "Plik nie jest poprawnym obrazem.") from None

    # GIF -> PNG (#51): płaskie kolory i alfa bez artefaktów JPEG.
    out_format = "PNG" if (keep_alpha or source_format == "GIF") else source_format
    quality = 85 if out_format == "JPEG" else None
    payload = _encode_image(resized, out_format, quality)

    # Step-down (rzadka ścieżka): normalizacja musi mieścić się w SEND_MAX_IMAGE_BYTES.
    # Kolejno: re-encode do JPEG, obniżanie quality, na końcu zmniejszanie wymiarów -
    # dzięki temu pętla realnie konwerguje zamiast zwracać oversized payload.
    attempts = 0
    while len(payload) > SEND_MAX_IMAGE_BYTES and attempts < 4:
        attempts += 1
        if out_format != "JPEG":
            resized = resized.convert("RGB")
            out_format, quality = "JPEG", 85
        elif (quality or 85) > SEND_MIN_QUALITY:
            quality = max(SEND_MIN_QUALITY, (quality or 85) - 15)
        else:
            side = max(256, max(resized.size) // 2)
            ratio = side / float(max(resized.size))
            resized = resized.resize(
                (max(1, int(resized.size[0] * ratio)), max(1, int(resized.size[1] * ratio))),
                Image.LANCZOS,
            )
        payload = _encode_image(resized, out_format, quality)

    return payload, ALLOWED_FORMATS[out_format][0], resized.size[0], resized.size[1]


# Referencja do załącznika w treści wiadomości user (flow.context trzyma ID, nie base64):
#   {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "attachment", "id": "..."}]}
# Zamiana na format API (image_url z data URL) dopiero przy budowaniu zapytania - resolve_messages().
ATTACHMENT_PART = "attachment"


def attachment_part(attachment: Attachment | str) -> dict:
    attachment_id = attachment.id if isinstance(attachment, Attachment) else attachment
    return {"type": ATTACHMENT_PART, "id": attachment_id}


def resolve_parts(content, store: "AttachmentStore", chat_id: str | None):
    """Zamienia referencje w treści na image_url z base64. Tekst i inne części bez zmian."""
    if not isinstance(content, list):
        return content
    resolved = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == ATTACHMENT_PART:
            attachment = store.get(chat_id, part.get("id"))
            encoded = base64.b64encode(store.read_bytes(chat_id, attachment.id)).decode("ascii")
            resolved.append({
                "type": "image_url",
                "image_url": {"url": f"data:{attachment.mime};base64,{encoded}"},
            })
        else:
            resolved.append(part)
    return resolved


def resolve_messages(messages: list[dict], store: "AttachmentStore", chat_id: str | None) -> list[dict]:
    """Kopia listy wiadomości gotowa do API - nie modyfikuje przekazanego kontekstu."""
    return [
        {**message, "content": resolve_parts(message["content"], store, chat_id)}
        if isinstance(message.get("content"), list) else message
        for message in messages
    ]


class AttachmentResolver:
    """Zamienia referencje załączników (ID) na image_url z data URL (#51).

    Wołany przez AgentFlow _tuż przed_ wysłaniem zapytania: flow.context trzyma
    wyłącznie ID, więc ani w history, ani u modelu nie lądują base64 ani ścieżki
    dysku. Normalizacja (prepare_image) i LRU cache data URL – powtórne iteracje
    tego samego runu nie dekodują megabajtów po raz drugi.
    """

    def __init__(
        self,
        store: "AttachmentStore",
        chat_id: str | None = None,
        *,
        max_side: int = SEND_MAX_SIDE,
        total_budget: int = SEND_TOTAL_BUDGET,
    ) -> None:
        self.store = store
        self.chat_id = chat_id
        self.max_side = int(max_side)
        self.total_budget = int(total_budget)
        self._cache: OrderedDict[str, str] = OrderedDict()

    def _data_url(self, attachment_id) -> tuple[str, int]:
        """Zwraca (data URL, jego długość) dla ID; wynik cache'owany per id (#51)."""
        if not isinstance(attachment_id, str):
            raise AttachmentError("not_found", "Nie znaleziono załącznika.")
        cached = self._cache.get(attachment_id)
        if cached is not None:
            self._cache.move_to_end(attachment_id)  # LRU touch
            return cached, len(cached)
        attachment = self.store.get(self.chat_id, attachment_id)
        data, mime, _, _ = prepare_image(
            self.store.read_bytes(self.chat_id, attachment_id), max_side=self.max_side
        )
        url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        self._cache[attachment_id] = url
        if len(self._cache) > RESOLVER_CACHE_SIZE:
            self._cache.popitem(last=False)
        return url, len(url)

    def __call__(self, messages: list[dict]) -> list[dict]:
        """Kopia `messages` z data URL; oryginalny kontekst NIE jest mutowany."""
        resolved: list[dict] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                parts = [dict(part) if isinstance(part, dict) else part for part in content]
                resolved.append({**message, "content": parts})
            else:
                resolved.append(message)

        # Budżet przydzielamy od końca - najnowszy turn obrazuje aktualny problem.
        remaining = self.total_budget
        for index in range(len(resolved) - 1, -1, -1):
            content = resolved[index].get("content")
            if not isinstance(content, list):
                continue
            for part_index in range(len(content) - 1, -1, -1):
                part = content[part_index]
                if not (isinstance(part, dict) and part.get("type") == ATTACHMENT_PART):
                    continue
                url, size = self._data_url(part.get("id"))
                if size <= remaining:
                    remaining -= size
                    content[part_index] = {"type": "image_url", "image_url": {"url": url}}
                else:
                    # Zamiast cicho ucinac - model dostaje jawną informację (#51).
                    content[part_index] = {
                        "type": "text",
                        "text": (
                            "[obraz pominięty w tym zapytaniu - przekroczony limit "
                            "wysyłki obrazów]"
                        ),
                    }
        return resolved


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Zapis załączników w plikach data/attachments/<chat_id|_pending>/<id>.*
class AttachmentStore:

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else DATA_DIR / "attachments"

    def _chat_dir(self, chat_id: str | None) -> Path:
        if chat_id is None:
            return self.directory / PENDING_DIR
        safe_id = Path(chat_id or "").name
        if not safe_id or safe_id != chat_id or safe_id.startswith("_"):
            raise AttachmentError("not_found", "Nie znaleziono załącznika.")
        return self.directory / safe_id

    def _meta_path(self, chat_id: str | None, attachment_id: str) -> Path:
        if not isinstance(attachment_id, str) or not _ID_RE.match(attachment_id):
            raise AttachmentError("not_found", "Nie znaleziono załącznika.")
        return self._chat_dir(chat_id) / f"{attachment_id}.json"

    def _write(self, attachment: Attachment, data: bytes) -> None:
        folder = self._chat_dir(attachment.chat_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{attachment.id}.{attachment.extension}").write_bytes(data)
        # metadane zapisywane na końcu - ich obecność oznacza kompletny załącznik
        (folder / f"{attachment.id}.json").write_text(
            json.dumps(attachment.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    def save(self, data: bytes, name: str = "", chat_id: str | None = None) -> Attachment:
        """Waliduje i zapisuje obraz. Bez chat_id trafia do oczekujących (_pending)."""
        mime, width, height = validate_image(data)
        self._chat_dir(chat_id)  # walidacja chat_id przed zapisem
        attachment = Attachment(
            id=uuid.uuid4().hex,
            mime=mime,
            size=len(data),
            width=width,
            height=height,
            chat_id=chat_id,
            created_at=_now(),
            # Nazwa tylko do wyświetlenia: odrzuwamy składowe katalogów, także
            # w stylu Windowsa ("C:\Users\x\zrzut.png"), gdzie Path() na POSIX
            # nie traktuje backslashu jako separatora.
            name=str(name or "").replace("\\", "/").rsplit("/", 1)[-1][:120],
        )
        self._write(attachment, data)
        return attachment

    def get(self, chat_id: str | None, attachment_id: str) -> Attachment:
        path = self._meta_path(chat_id, attachment_id)
        if not path.is_file():
            raise AttachmentError("not_found", "Nie znaleziono załącznika.")
        return Attachment.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def read_bytes(self, chat_id: str | None, attachment_id: str) -> bytes:
        attachment = self.get(chat_id, attachment_id)
        return (self._chat_dir(chat_id) / f"{attachment.id}.{attachment.extension}").read_bytes()

    def list_for_chat(self, chat_id: str | None) -> list[Attachment]:
        """Metadane załączników czatu (najnowsze first) - tylko metadata, bez bytes."""
        folder = self._chat_dir(chat_id)
        attachments: list[Attachment] = []
        if not folder.is_dir():
            return attachments
        for meta in folder.glob("*.json"):
            try:
                attachments.append(
                    Attachment.from_dict(json.loads(meta.read_text(encoding="utf-8")))
                )
            except (OSError, ValueError, TypeError, KeyError):
                continue
        attachments.sort(key=lambda item: item.created_at, reverse=True)
        return attachments

    def bind(self, attachment_ids: list[str], chat_id: str) -> list[Attachment]:
        """Przenosi oczekujące załączniki do czatu. Już przypisane do tego czatu są pomijane."""
        if chat_id is None:
            raise AttachmentError("not_found", "Brak czatu do przypisania załączników.")
        target = self._chat_dir(chat_id)
        bound = []
        for attachment_id in attachment_ids:
            if self._meta_path(chat_id, attachment_id).is_file():
                bound.append(self.get(chat_id, attachment_id))
                continue
            pending = self.get(None, attachment_id)
            source = self._chat_dir(None)
            target.mkdir(parents=True, exist_ok=True)
            file_name = f"{pending.id}.{pending.extension}"
            (source / file_name).replace(target / file_name)
            attachment = Attachment.from_dict({**pending.to_dict(), "chat_id": chat_id})
            (target / f"{pending.id}.json").write_text(
                json.dumps(attachment.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            (source / f"{pending.id}.json").unlink()
            bound.append(attachment)
        return bound

    def delete(self, chat_id: str | None, attachment_id: str) -> bool:
        try:
            attachment = self.get(chat_id, attachment_id)
        except AttachmentError:
            return False
        folder = self._chat_dir(chat_id)
        (folder / f"{attachment.id}.json").unlink(missing_ok=True)
        (folder / f"{attachment.id}.{attachment.extension}").unlink(missing_ok=True)
        return True

    def delete_for_chat(self, chat_id: str) -> None:
        """Usuwa wszystkie załączniki czatu - wołać przy ChatStore.delete i ChatStore.cleanup."""
        folder = self._chat_dir(chat_id)
        if folder.is_dir():
            shutil.rmtree(folder)

    def cleanup_pending(self, max_age_hours: int = PENDING_TTL_HOURS) -> list[str]:
        """Usuwa oczekujące załączniki starsze niż max_age_hours (nigdy niewysłane)."""
        folder = self._chat_dir(None)
        removed: list[str] = []
        if not folder.is_dir():
            return removed
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        for meta in folder.glob("*.json"):
            try:
                attachment = Attachment.from_dict(json.loads(meta.read_text(encoding="utf-8")))
                expired = datetime.fromisoformat(attachment.created_at) < cutoff
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if expired and self.delete(None, attachment.id):
                removed.append(attachment.id)
        return removed
