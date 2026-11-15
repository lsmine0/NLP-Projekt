"""Pomocnicze funkcje do konwersji narzędzi MCP na format ToolHandle."""

from __future__ import annotations

import hashlib
import re
from typing import Any

MAX_TOOL_NAME_LENGTH = 64
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")

# Keys, które rozumie OpenAI/Ollama tool calling. Reszta jest odrzucana, żeby
# nie wysyłać modelowi schematu, którego nie potrafi zinterpretować.
_ALLOWED_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "description",
    "title",
    "default",
    "enum",
    "format",
    "items",
    "additionalProperties",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "anyOf",
    "oneOf",
    "allOf",
}

_DROPPED_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "$anchor",
    "$comment",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$defs",
    "definitions",
}


def _sanitize_name_part(value: str, fallback: str) -> str:
    cleaned = _NAME_RE.sub("_", str(value or "").strip()).strip("_")
    return cleaned or fallback


def exposed_tool_name(
    server_name: str,
    tool_name: str,
    existing: set[str] | None = None,
) -> str:
    """Zwraca nazwę narzędzia widoczną dla modelu: ``mcp_<serwer>_<tool>``.

    Nazwy są ograniczone do znaków akceptowanych przez OpenAI/Ollama.
    W razie kolizji dodawany jest krótki hash oryginalnej pary.
    """

    existing = existing or set()
    base = (
        f"mcp_{_sanitize_name_part(server_name, 'server')}_"
        f"{_sanitize_name_part(tool_name, 'tool')}"
    )
    base = base[:MAX_TOOL_NAME_LENGTH]
    if base not in existing:
        return base

    digest = hashlib.sha1(f"{server_name}/{tool_name}".encode("utf-8")).hexdigest()[:6]
    candidate = f"{base[: MAX_TOOL_NAME_LENGTH - 7]}_{digest}"
    if candidate not in existing:
        return candidate

    for counter in range(2, 1000):
        suffix = f"_{digest}_{counter}"
        candidate = f"{base[: MAX_TOOL_NAME_LENGTH - len(suffix)]}{suffix}"
        if candidate not in existing:
            return candidate

    raise ValueError(f"Nie udało się utworzyć unikalnej nazwy dla '{server_name}/{tool_name}'.")


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None

    node: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit():
            index = int(part)
            if index >= len(node):
                return None
            node = node[index]
        else:
            return None

    return node if isinstance(node, dict) else None


def sanitize_schema(
    schema: Any,
    *,
    root: dict[str, Any] | None = None,
    depth: int = 0,
) -> dict[str, Any]:
    """Tworzy konserwatywny podzbiór JSON Schema akceptowany przez Ollamę.

    MCP używa pełnego JSON Schema. Część serwerów generuje ``$ref``, ``$defs``
    albo ``anyOf``, których Ollama może nie obsłużyć. Funkcja rozwiązuje lokalne
    referencje i zostawia tylko najczęściej wspierane słowa kluczowe.
    """

    if depth > 20 or not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}

    if root is None:
        root = schema

    ref = schema.get("$ref")
    if isinstance(ref, str):
        resolved = _resolve_ref(ref, root)
        if resolved is not None:
            merged = dict(resolved)
            merged.update({key: value for key, value in schema.items() if key != "$ref"})
            return sanitize_schema(merged, root=root, depth=depth + 1)

    result: dict[str, Any] = {}

    for key, value in schema.items():
        if key in _DROPPED_SCHEMA_KEYS:
            continue

        if key == "properties" and isinstance(value, dict):
            result["properties"] = {
                str(name): sanitize_schema(item, root=root, depth=depth + 1)
                for name, item in value.items()
                if isinstance(item, dict)
            }
        elif key == "items":
            if isinstance(value, dict):
                result["items"] = sanitize_schema(value, root=root, depth=depth + 1)
            elif isinstance(value, list):
                result["items"] = [
                    sanitize_schema(item, root=root, depth=depth + 1)
                    for item in value
                    if isinstance(item, dict)
                ]
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [
                sanitize_schema(item, root=root, depth=depth + 1)
                for item in value
                if isinstance(item, dict)
            ]
        elif key == "additionalProperties":
            if isinstance(value, dict):
                result[key] = sanitize_schema(value, root=root, depth=depth + 1)
            elif isinstance(value, bool):
                result[key] = value
        elif key == "required" and isinstance(value, list):
            result[key] = [str(item) for item in value if isinstance(item, str)]
        elif key in _ALLOWED_SCHEMA_KEYS:
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            elif isinstance(value, list) and all(
                isinstance(item, (str, int, float, bool)) for item in value
            ):
                result[key] = value

    schema_type = result.get("type")
    if isinstance(schema_type, list) and "object" in schema_type:
        result["type"] = "object"
        schema_type = "object"
    if not schema_type:
        if "properties" in result:
            result["type"] = "object"
        elif "items" in result:
            result["type"] = "array"
        elif isinstance(result.get("enum"), list) and result["enum"]:
            first = result["enum"][0]
            if isinstance(first, bool):
                result["type"] = "boolean"
            elif isinstance(first, int):
                result["type"] = "integer"
            elif isinstance(first, float):
                result["type"] = "number"
            elif isinstance(first, str):
                result["type"] = "string"
        elif depth == 0:
            result["type"] = "object"

    if result.get("type") == "object":
        properties = result.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            result["properties"] = properties
        required = result.get("required")
        if not isinstance(required, list):
            required = []
        result["required"] = [name for name in required if name in properties]

    return result


def sanitize_tool_input_schema(schema: Any) -> dict[str, Any]:
    """Zwraca schemat parametrów narzędzia w formacie oczekiwanym przez ToolHandle."""

    sanitized = sanitize_schema(schema)
    if sanitized.get("type") != "object":
        # Zgodnie ze specyfikacją MCP inputSchema powinien opisywać obiekt.
        return {"type": "object", "properties": {}, "required": []}
    sanitized.setdefault("properties", {})
    sanitized.setdefault("required", [])
    return sanitized
