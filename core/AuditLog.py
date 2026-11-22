"""Append-only audit log for tool decisions and destructive operations.

The log deliberately accepts only a whitelist of metadata fields. Arguments,
tool output and file content are never written here.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import Logger

_ALLOWED_FIELDS = frozenset(
    {
        "tool",
        "risk",
        "decision",
        "reason",
        "approval_id",
        "chat_id",
        "run_id",
        "source",
        "phase",
        "success",
        "duration_ms",
        "policy",
        "code",
        "kind",
    }
)


class AuditLog:
    """Thread-safe JSONL audit writer.

    ``path=None`` disables auditing (useful for unit tests and library use).
    A write failure is reported but never interrupts the agent turn.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._lock = threading.Lock()

    def _write(self, entry: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # os.open with mode 0600 avoids a brief world-readable window
                # between file creation and a separate chmod call.
                descriptor = os.open(
                    self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
            except OSError as exc:
                Logger.printWarning(f"Nie można zapisać dziennika audytu: {exc}")

    def record(self, kind: str, **fields: Any) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind),
        }
        for key, value in fields.items():
            if key not in _ALLOWED_FIELDS or value is None:
                continue
            if key == "risk":
                value = [value] if isinstance(value, str) else [str(item) for item in value]
            elif isinstance(value, (list, tuple, set, frozenset)):
                value = [
                    item for item in value
                    if isinstance(item, (str, int, float, bool))
                ]
            elif not isinstance(value, (str, int, float, bool)):
                # Defensive: never let an accidental dict/object leak arguments.
                continue
            entry[key] = value
        self._write(entry)

    def decision(
        self,
        *,
        tool: str,
        risk,
        decision: str,
        reason: str,
        chat_id: str | None = None,
        run_id: str | None = None,
        approval_id: str | None = None,
        source: str | None = None,
        policy: str | None = None,
        code: str | None = None,
    ) -> None:
        self.record(
            "decision",
            tool=tool,
            risk=risk,
            decision=decision,
            reason=reason,
            chat_id=chat_id,
            run_id=run_id,
            approval_id=approval_id,
            source=source,
            policy=policy,
            code=code,
        )

    def destructive(
        self,
        *,
        tool: str,
        risk,
        phase: str,
        chat_id: str | None = None,
        run_id: str | None = None,
        approval_id: str | None = None,
        success: bool | None = None,
        duration_ms: int | None = None,
        decision: str | None = None,
    ) -> None:
        self.record(
            "destructive",
            tool=tool,
            risk=risk,
            phase=phase,
            chat_id=chat_id,
            run_id=run_id,
            approval_id=approval_id,
            success=success,
            duration_ms=duration_ms,
            decision=decision,
        )

    def close(self) -> None:
        # File handles are opened per append, so there is nothing to flush.
        return None
