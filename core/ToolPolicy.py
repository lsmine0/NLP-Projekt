"""Risk labels, allow/ask/deny policy and path-safety helpers for tools.

The module is deliberately free of GUI/CLI dependencies so the registry can be
used by the headless API, the desktop app and tests alike.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

RISK_LABELS = ("read", "write", "destructive", "external")
RISK_SET = frozenset(RISK_LABELS)
POLICIES = ("allow", "ask", "deny")
POLICY_RANK = {"allow": 0, "ask": 1, "deny": 2}

# Safe defaults: reads are harmless, everything that can change state, destroy
# data, execute code or touch the network asks the user first.
DEFAULT_RISK_POLICIES: dict[str, str] = {
    "read": "allow",
    "write": "ask",
    "destructive": "ask",
    "external": "ask",
}

TOOL_KINDS = ("tool", "user_question")

_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|token|secret|password|passwd|authorization|cookie|"
    r"credential|private[_-]?key|session[_-]?id|access[_-]?key)",
    re.IGNORECASE,
)

_SECRET_VALUE_RES = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|token|secret|password|passwd)['\"]?\s*[:=]\s*['\"]?[^\s'\"]+"
    ),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
    re.compile(r"(?i)\bhttps?://[^/\s:@]+:[^/\s@]+@"),
)

_MAX_SAFE_STRING = 300
_MAX_SAFE_ITEMS = 25
_MAX_SAFE_DEPTH = 4
_CONTENT_KEYS = frozenset({"content", "code", "replacement", "target"})


def normalize_risk(value: str | Iterable[str] | None) -> frozenset[str]:
    """Normalize one or more risk labels; labels are intentionally combinable."""

    if value is None:
        return frozenset()
    if isinstance(value, str):
        parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    else:
        try:
            parts = list(value)
        except TypeError as exc:
            raise ValueError("Risk labels must be a string or an iterable of strings.") from exc

    labels: set[str] = set()
    for part in parts:
        if not isinstance(part, str):
            raise ValueError("Risk labels must be strings.")
        label = part.strip().lower()
        if not label:
            continue
        if label not in RISK_SET:
            allowed = ", ".join(RISK_LABELS)
            raise ValueError(f"Unknown risk label '{part}'. Allowed labels: {allowed}.")
        labels.add(label)
    # Keep a deterministic order for events, logs and UI badges.
    return frozenset(label for label in RISK_LABELS if label in labels)


def normalize_policy(value: str | None, *, default: str | None = None) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if not isinstance(value, str):
        raise ValueError("Policy must be one of: allow, ask, deny.")
    policy = value.strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy '{value}'. Allowed policies: allow, ask, deny.")
    return policy


def most_restrictive_policy(policies: Iterable[str | None]) -> str | None:
    resolved = [policy for policy in policies if policy]
    if not resolved:
        return None
    return max(resolved, key=lambda item: POLICY_RANK.get(item, 0))


def normalize_roots(roots: Iterable[str | Path] | None) -> tuple[Path, ...]:
    """Resolve allowed directories and require them to exist."""

    normalized: list[Path] = []
    for raw in roots or ():
        if isinstance(raw, bool) or not isinstance(raw, (str, Path)):
            raise ValueError("Allowed directory must be a string or Path.")
        if isinstance(raw, str) and not raw.strip():
            raise ValueError("Allowed directory path cannot be empty.")
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise ValueError(f"Invalid allowed directory: {path}") from exc
        if path.exists() and not resolved.is_dir():
            raise ValueError(f"Allowed directory is not a directory: {path}")
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def normalize_protected_paths(paths: Iterable[str | Path] | None) -> tuple[Path, ...]:
    """Resolve protected paths; they may not exist yet (e.g. audit log)."""

    normalized: list[Path] = []
    for raw in paths or ():
        if isinstance(raw, bool) or not isinstance(raw, (str, Path)):
            raise ValueError("Protected path must be a string or Path.")
        if isinstance(raw, str) and not raw.strip():
            continue
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            continue
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def is_protected(path: Path, protected_paths: Iterable[Path]) -> bool:
    for protected in protected_paths:
        if path == protected or protected in path.parents:
            return True
    return False


def normalize_tool_path(
    value: Any,
    *,
    allowed_roots: Iterable[Path],
    primary_root: Path | None,
) -> Path:
    """Normalize a model-provided path and verify it stays inside allowed roots.

    Relative paths are interpreted against ``primary_root`` (the workspace).
    ``resolve(strict=False)`` normalizes ``..`` and follows existing symlinks,
    so a symlink pointing outside the workspace is rejected.
    """

    if not isinstance(value, (str, Path)):
        raise ValueError("Path must be a string or Path.")
    text = str(value).strip()
    if not text:
        raise ValueError("Path cannot be empty.")
    if "\x00" in text:
        raise ValueError("Path contains a null byte.")

    roots = tuple(allowed_roots)
    if not roots:
        return Path(text).expanduser().resolve(strict=False)

    raw = Path(text).expanduser()
    base = primary_root or roots[0]
    candidate = raw if raw.is_absolute() else (base / raw)
    resolved = candidate.resolve(strict=False)

    if not any(resolved.is_relative_to(root) for root in roots):
        raise PermissionError("Ścieżka wykracza poza dozwolone katalogi robocze.")
    return resolved


def redact_text(value: str) -> str:
    text = value
    for pattern in _SECRET_VALUE_RES:
        text = pattern.sub("***", text)
    return text


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, str):
        redacted = redact_text(value)
        if len(redacted) > _MAX_SAFE_STRING:
            return redacted[:_MAX_SAFE_STRING] + f"… ({len(redacted)} znaków)"
        return redacted
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))


def _safe_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > _MAX_SAFE_DEPTH:
        return "…"
    if _SENSITIVE_KEY_RE.search(key):
        return "***"
    if key in _CONTENT_KEYS and isinstance(value, str):
        return {
            "length": len(value),
            "lines": value.count("\n") + 1 if value else 0,
        }
    if isinstance(value, Mapping):
        return {
            str(name): _safe_value(item, key=str(name), depth=depth + 1)
            for name, item in list(value.items())[:_MAX_SAFE_ITEMS]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _safe_value(item, key=key, depth=depth + 1)
            for item in list(value)[:_MAX_SAFE_ITEMS]
        ]
    return _safe_scalar(value)


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return redact_text(value)
    if not parts.scheme or not parts.netloc:
        return redact_text(value)
    query = [
        (name, "***" if _SENSITIVE_KEY_RE.search(name) else redact_text(item))
        for name, item in parse_qsl(parts.query, keep_blank_values=True)
    ]
    host = parts.hostname or ""
    netloc = host
    try:
        port = parts.port
    except ValueError:
        port = None
    if port:
        netloc = f"{host}:{port}"
    return urlunsplit(
        (parts.scheme, netloc, redact_text(parts.path), urlencode(query), "")
    )


def safe_arguments(spec: Any, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a redacted, bounded argument preview suitable for GUI/CLI.

    Execution always uses the original arguments; this mapping is presentation
    only. File content/code is replaced with a redacted, short preview so that
    accidental secrets do not reach the approval dialog or audit trail.
    """

    if not isinstance(arguments, Mapping):
        return {}

    result: dict[str, Any] = {}
    for raw_key, value in list(arguments.items())[:_MAX_SAFE_ITEMS]:
        key = str(raw_key)
        if _SENSITIVE_KEY_RE.search(key):
            result[key] = "***"
            continue
        if key in _CONTENT_KEYS and isinstance(value, str):
            # Never copy file content or generated code into the approval dialog:
            # arbitrary secrets cannot be reliably detected by a regex.
            result[key] = {
                "length": len(value),
                "lines": value.count("\n") + 1 if value else 0,
            }
            continue
        if key in {"url", "uri"} and isinstance(value, str):
            result[key] = _safe_url(value)
            continue
        if key in {"path", "source", "destination"} and isinstance(value, (str, Path)):
            result[key] = redact_text(str(value))
            continue
        result[key] = _safe_value(value, key=key)
    return result


def format_safe_arguments(spec: Any, arguments: Mapping[str, Any] | None) -> str:
    return json.dumps(
        safe_arguments(spec, arguments), ensure_ascii=False, indent=2, default=str
    )


@dataclass(frozen=True)
class ToolSpec:
    """Metadata stored next to a tool schema, never sent to the model."""

    name: str
    risk: frozenset[str] = field(default_factory=frozenset)
    policy: str | None = None
    source: str = "builtin"
    server: str | None = None
    remote_name: str | None = None
    path_args: tuple[str, ...] = ()
    kind: str = "tool"
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in TOOL_KINDS:
            raise ValueError(f"Unknown tool kind '{self.kind}'.")
        if self.source not in ("builtin", "mcp"):
            raise ValueError(f"Unknown tool source '{self.source}'.")
        object.__setattr__(self, "risk", normalize_risk(self.risk))
        object.__setattr__(
            self,
            "policy",
            normalize_policy(self.policy),
        )
        if self.path_args is None:
            object.__setattr__(self, "path_args", ())
        elif isinstance(self.path_args, str):
            object.__setattr__(self, "path_args", (self.path_args,))
        else:
            object.__setattr__(self, "path_args", tuple(str(item) for item in self.path_args))

    @property
    def destructive(self) -> bool:
        return "destructive" in self.risk

    @property
    def external(self) -> bool:
        return "external" in self.risk

    def public(self, policy: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk": [label for label in RISK_LABELS if label in self.risk],
            "policy": policy,
            "source": self.source,
            "server": self.server,
            "remote_name": self.remote_name,
            "kind": self.kind,
        }


@dataclass
class PreparedToolCall:
    name: str
    arguments: dict[str, Any]
    spec: ToolSpec
    policy: str
    requires_approval: bool = False
    denial: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.denial is not None

    @property
    def risk(self) -> frozenset[str]:
        return self.spec.risk
