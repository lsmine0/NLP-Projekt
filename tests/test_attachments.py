import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import Attachments
from core.Attachments import AttachmentError, AttachmentStore, validate_image
from core.ChatStore import ChatStore

CHAT = "a" * 32
OTHER_CHAT = "b" * 32


def image_bytes(fmt="PNG", size=(4, 3)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    return AttachmentStore(tmp_path)


@pytest.mark.parametrize("fmt, mime", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("GIF", "image/gif"),
])
def test_validate_accepts_allowed_formats(fmt, mime):
    assert validate_image(image_bytes(fmt)) == (mime, 4, 3)


@pytest.mark.parametrize("data, code", [
    (b"", "empty"),
    (b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", "bad_format"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "bad_format"),
])
def test_validate_rejects_bad_input(data, code):
    with pytest.raises(AttachmentError) as exc:
        validate_image(data)
    assert exc.value.code == code


def test_validate_rejects_disallowed_format():
    with pytest.raises(AttachmentError) as exc:
        validate_image(image_bytes("BMP"))
    assert exc.value.code == "bad_format"


def test_validate_limits(monkeypatch):
    monkeypatch.setattr(Attachments, "MAX_BYTES", 10)
    with pytest.raises(AttachmentError) as exc:
        validate_image(image_bytes())
    assert exc.value.code == "too_large"

    monkeypatch.setattr(Attachments, "MAX_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(Attachments, "MAX_PIXELS", 11)
    with pytest.raises(AttachmentError) as exc:
        validate_image(image_bytes())
    assert exc.value.code == "too_large"


def test_save_pending_then_bind(store, tmp_path):
    data = image_bytes()
    pending = store.save(data, name="C:\\Users\\ktoś\\zrzut.png")
    assert pending.chat_id is None
    assert pending.name == "zrzut.png"
    assert store.read_bytes(None, pending.id) == data

    [bound] = store.bind([pending.id], CHAT)
    assert bound.chat_id == CHAT
    assert store.read_bytes(CHAT, pending.id) == data
    with pytest.raises(AttachmentError):
        store.get(None, pending.id)
    # ponowne przypisanie do tego samego czatu jest bezpieczne
    assert store.bind([pending.id], CHAT) == [bound]
    assert not list((tmp_path / Attachments.PENDING_DIR).iterdir())


def test_access_is_scoped_to_chat(store):
    attachment = store.save(image_bytes(), chat_id=CHAT)
    with pytest.raises(AttachmentError) as exc:
        store.get(OTHER_CHAT, attachment.id)
    assert exc.value.code == "not_found"
    with pytest.raises(AttachmentError):
        store.bind([attachment.id], OTHER_CHAT)


@pytest.mark.parametrize("chat_id, attachment_id", [
    ("../x", "c" * 32),
    ("_pending", "c" * 32),
    (CHAT, "../../settings"),
    (CHAT, "C" * 32),
])
def test_rejects_unsafe_ids(store, chat_id, attachment_id):
    with pytest.raises(AttachmentError) as exc:
        store.get(chat_id, attachment_id)
    assert exc.value.code == "not_found"


def test_delete_and_delete_for_chat(store, tmp_path):
    first = store.save(image_bytes(), chat_id=CHAT)
    second = store.save(image_bytes(), chat_id=CHAT)
    assert store.delete(CHAT, first.id)
    assert not store.delete(CHAT, first.id)
    store.get(CHAT, second.id)

    store.delete_for_chat(CHAT)
    assert not (tmp_path / CHAT).exists()
    store.delete_for_chat(CHAT)  # brak folderu to nie błąd


def test_cleanup_pending_removes_only_expired(store, tmp_path):
    old = store.save(image_bytes())
    fresh = store.save(image_bytes())
    meta = tmp_path / Attachments.PENDING_DIR / f"{old.id}.json"
    record = json.loads(meta.read_text(encoding="utf-8"))
    record["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    meta.write_text(json.dumps(record), encoding="utf-8")

    assert store.cleanup_pending() == [old.id]
    store.get(None, fresh.id)
    assert not list((tmp_path / Attachments.PENDING_DIR).glob(f"{old.id}.*"))


def test_chat_store_keeps_attachment_ids(tmp_path):
    chats = ChatStore(tmp_path)
    chat = chats.create()
    chats.append(chat["id"], "user", "co jest na zdjęciu?", attachments=["c" * 32])
    chats.append(chat["id"], "assistant", "kot")
    messages = chats.get(chat["id"])["messages"]
    assert messages[0]["attachments"] == ["c" * 32]
    assert "attachments" not in messages[1]
