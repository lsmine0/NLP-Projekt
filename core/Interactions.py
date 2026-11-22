"""Pending approvals and agent questions shared by GUI, CLI and API."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

APPROVAL_DECISIONS = ("allow_once", "allow_session", "deny")
POLL_PENDING = "pending"
POLL_TIMEOUT = "timeout"
POLL_CANCELLED = "cancelled"


@dataclass
class PendingInteraction:
    id: str
    kind: str  # "approval" | "question"
    chat_id: str | None
    run_id: str | None
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    tool: str | None = None
    risk: tuple[str, ...] = ()
    decision: str | None = None
    answer: dict[str, Any] | None = None
    cancelled: bool = False
    resolved: bool = False

    @property
    def expires_in(self) -> int:
        return max(0, int(round(self.expires_at - time.monotonic())))


@dataclass
class ResolveResult:
    ok: bool
    code: str
    pending: PendingInteraction | None = None


class InteractionBroker:
    """In-memory broker with short polling instead of blocking waits.

    Polling lets ``AgentFlow`` yield heartbeat events while waiting, so the GUI
    can keep checking for a disconnected client and cancel the run.
    """

    def __init__(self, timeout: float = 300.0) -> None:
        self.timeout = max(0.01, float(timeout))
        self._pending: dict[str, PendingInteraction] = {}
        self._session_allows: set[tuple[str | None, str]] = set()
        self._cancelled_runs: set[str] = set()
        self._cancelled_chats: set[str] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex

    def register_approval(
        self,
        *,
        tool: str,
        risk,
        arguments: dict[str, Any],
        chat_id: str | None = None,
        run_id: str | None = None,
        source: str = "builtin",
        server: str | None = None,
        expires_in: float | None = None,
    ) -> PendingInteraction:
        now = time.monotonic()
        pending = PendingInteraction(
            id=f"apr_{uuid.uuid4().hex}",
            kind="approval",
            chat_id=chat_id,
            run_id=run_id,
            payload={
                "tool": tool,
                "source": source,
                "server": server,
                "risk": list(risk),
                "arguments": dict(arguments),
            },
            created_at=now,
            expires_at=now + (float(expires_in) if expires_in is not None else self.timeout),
            tool=tool,
            risk=tuple(risk),
        )
        with self._lock:
            self._pending[pending.id] = pending
        return pending

    def register_question(
        self,
        *,
        question: str,
        options: list[str],
        header: str = "",
        allow_custom: bool = True,
        chat_id: str | None = None,
        run_id: str | None = None,
        expires_in: float | None = None,
    ) -> PendingInteraction:
        now = time.monotonic()
        pending = PendingInteraction(
            id=f"ask_{uuid.uuid4().hex}",
            kind="question",
            chat_id=chat_id,
            run_id=run_id,
            payload={
                "question": question,
                "options": list(options),
                "header": header,
                "allow_custom": bool(allow_custom),
            },
            created_at=now,
            expires_at=now + (float(expires_in) if expires_in is not None else self.timeout),
        )
        with self._lock:
            self._pending[pending.id] = pending
        return pending

    # ------------------------------------------------------------------
    # Session-scoped permissions
    # ------------------------------------------------------------------
    def session_allowed(self, chat_id: str | None, tool: str) -> bool:
        with self._lock:
            return (chat_id, tool) in self._session_allows

    def allow_session(self, chat_id: str | None, tool: str) -> None:
        with self._lock:
            self._session_allows.add((chat_id, tool))

    def forget_chat(self, chat_id: str) -> None:
        with self._lock:
            self._session_allows = {
                item for item in self._session_allows if item[0] != chat_id
            }
            for pending_id, pending in list(self._pending.items()):
                if pending.chat_id == chat_id:
                    pending.cancelled = True
                    self._pending.pop(pending_id, None)
                    self._forget_run_if_idle(pending.run_id)
            self._cancelled_chats.discard(chat_id)

    def clear_session_allows(self) -> None:
        with self._lock:
            self._session_allows.clear()

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve_approval(self, approval_id: str, decision: str) -> ResolveResult:
        if decision not in APPROVAL_DECISIONS:
            return ResolveResult(False, "invalid_decision")
        now = time.monotonic()
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or pending.kind != "approval":
                return ResolveResult(False, "not_found")
            if pending.resolved:
                return ResolveResult(False, "already_resolved", pending)
            if pending.cancelled:
                return ResolveResult(False, "cancelled", pending)
            if now >= pending.expires_at:
                pending.resolved = True
                return ResolveResult(False, "expired", pending)

            pending.decision = decision
            pending.resolved = True
            if decision == "allow_session" and pending.tool:
                self._session_allows.add((pending.chat_id, pending.tool))
            return ResolveResult(True, "ok", pending)

    def resolve_question(self, question_id: str, answer: dict[str, Any]) -> ResolveResult:
        now = time.monotonic()
        with self._lock:
            pending = self._pending.get(question_id)
            if pending is None or pending.kind != "question":
                return ResolveResult(False, "not_found")
            if pending.resolved:
                return ResolveResult(False, "already_resolved", pending)
            if pending.cancelled:
                return ResolveResult(False, "cancelled", pending)
            if now >= pending.expires_at:
                pending.resolved = True
                return ResolveResult(False, "expired", pending)
            pending.answer = dict(answer)
            pending.resolved = True
            return ResolveResult(True, "ok", pending)

    def poll(self, pending: PendingInteraction) -> str | dict[str, Any] | None:
        """Return a decision/answer, a terminal state, or ``"pending"``.

        A successful resolution is consumed exactly once. Expired and cancelled
        requests are removed and returned as terminal states.
        """

        now = time.monotonic()
        with self._lock:
            current = self._pending.get(pending.id)
            if current is None:
                if pending.cancelled:
                    return POLL_CANCELLED
                return POLL_TIMEOUT
            if current.cancelled:
                self._pending.pop(current.id, None)
                self._forget_run_if_idle(current.run_id)
                return POLL_CANCELLED
            if now >= current.expires_at and not current.resolved:
                current.resolved = True
                self._pending.pop(current.id, None)
                self._forget_run_if_idle(current.run_id)
                return POLL_TIMEOUT
            if current.resolved:
                self._pending.pop(current.id, None)
                self._forget_run_if_idle(current.run_id)
                if current.kind == "approval":
                    return current.decision
                return dict(current.answer or {"cancelled": True})
            if current.run_id and current.run_id in self._cancelled_runs:
                current.cancelled = True
                self._pending.pop(current.id, None)
                self._forget_run_if_idle(current.run_id)
                return POLL_CANCELLED
            if current.chat_id and current.chat_id in self._cancelled_chats:
                current.cancelled = True
                self._pending.pop(current.id, None)
                return POLL_CANCELLED
            return POLL_PENDING

    # ------------------------------------------------------------------
    # Cancellation / cleanup
    # ------------------------------------------------------------------
    def _forget_run_if_idle(self, run_id: str | None) -> None:
        if not run_id:
            return
        if not any(pending.run_id == run_id for pending in self._pending.values()):
            self._cancelled_runs.discard(run_id)

    def cancel_run(self, run_id: str | None) -> None:
        if not run_id:
            return
        with self._lock:
            self._cancelled_runs.add(run_id)
            for pending in self._pending.values():
                if pending.run_id == run_id:
                    pending.cancelled = True
            self._forget_run_if_idle(run_id)

    def cancel_chat(self, chat_id: str) -> None:
        with self._lock:
            self._cancelled_chats.add(chat_id)
            for pending in self._pending.values():
                if pending.chat_id == chat_id:
                    pending.cancelled = True

    def cancel_all_pending(self) -> None:
        with self._lock:
            for pending in self._pending.values():
                pending.cancelled = True
            self._pending.clear()
            self._cancelled_runs.clear()
            self._cancelled_chats.clear()

    def clear_run(self, run_id: str | None) -> None:
        if not run_id:
            return
        with self._lock:
            self._cancelled_runs.discard(run_id)

    def pending(self, interaction_id: str) -> PendingInteraction | None:
        with self._lock:
            return self._pending.get(interaction_id)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        with self._lock:
            for pending in self._pending.values():
                pending.cancelled = True
            self._pending.clear()
            self._session_allows.clear()
            self._cancelled_runs.clear()
            self._cancelled_chats.clear()


def public_approval_event(pending: PendingInteraction) -> dict[str, Any]:
    payload = pending.payload
    return {
        "type": "approval_required",
        "id": pending.id,
        "tool": payload.get("tool"),
        "source": payload.get("source"),
        "server": payload.get("server"),
        "risk": list(payload.get("risk") or []),
        "arguments": dict(payload.get("arguments") or {}),
        "expires_in": pending.expires_in,
    }


def public_question_event(pending: PendingInteraction) -> dict[str, Any]:
    payload = pending.payload
    return {
        "type": "user_input_required",
        "id": pending.id,
        "question": payload.get("question"),
        "options": list(payload.get("options") or []),
        "header": payload.get("header") or "",
        "allow_custom": bool(payload.get("allow_custom", True)),
        "expires_in": pending.expires_in,
    }
