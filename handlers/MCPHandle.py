"""Menedżer serwerów MCP i rejestracja ich narzędzi w ToolHandle."""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters

from core import Logger
from handlers.MCPSchema import exposed_tool_name, sanitize_tool_input_schema
from handlers.MCPRuntime import (
    MCPConfigError,
    MCPConnectionError,
    MCPRuntime,
    MCPToolError,
)
from handlers.ToolHandle import ToolHandle


DEFAULT_MCP_TIMEOUT = 8.0
DEFAULT_MCP_STARTUP_TIMEOUT = 20.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str] | None
    cwd: str | None
    timeout_s: float
    startup_timeout_s: float
    enabled: bool
    allow_all_tools: bool
    allow_tools: frozenset[str]
    deny_tools: frozenset[str]


def _expand(value: str) -> str:
    """Rozwija ``${PYTHON}``, ``${PROJECT_ROOT}``, zmienne środowiskowe i ``~``."""

    value = value.replace("${PYTHON}", sys.executable)
    value = value.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
    return os.path.expandvars(os.path.expanduser(value))


def _as_string_list(value: Any, field: str, server_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MCPConfigError(
            f"MCP '{server_name}': pole '{field}' musi być listą napisów."
        )
    return [item.strip() for item in value if item.strip()]


def _parse_tools(raw_tools: Any, server_name: str) -> tuple[
    bool, frozenset[str], frozenset[str]
]:
    if raw_tools is None:
        raw_tools = {}

    if isinstance(raw_tools, list):
        raw_tools = {"allow": raw_tools}
    if not isinstance(raw_tools, dict):
        raise MCPConfigError(
            f"MCP '{server_name}': pole 'tools' musi być obiektem lub listą."
        )

    allow_raw = raw_tools.get("allow", [])
    allow_all = allow_raw == "*" or (
        isinstance(allow_raw, list) and "*" in allow_raw
    )
    allow_tools = (
        frozenset()
        if allow_all
        else frozenset(_as_string_list(allow_raw, "tools.allow", server_name))
    )
    deny_tools = set(
        _as_string_list(raw_tools.get("deny", []), "tools.deny", server_name)
    )

    ask_tools = _as_string_list(
        raw_tools.get("ask", []), "tools.ask", server_name
    )
    if ask_tools:
        Logger.printWarning(
            f"MCP '{server_name}': polityka 'ask' nie jest jeszcze obsługiwana; "
            "narzędzia z listy 'ask' zostaną zablokowane."
        )
        deny_tools.update(ask_tools)

    return allow_all, allow_tools, frozenset(deny_tools)


def load_servers_config(path: str | Path | None) -> dict[str, MCPServerConfig]:
    """Wczytuje i waliduje plik ``mcp_servers.json``."""

    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        return {}

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MCPConfigError(
            f"Nie można wczytać konfiguracji MCP '{config_path}': {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise MCPConfigError("Konfiguracja MCP musi być obiektem JSON.")

    servers_raw = raw.get("servers", raw.get("mcpServers", {}))
    if not isinstance(servers_raw, dict):
        raise MCPConfigError("Pole 'servers' w konfiguracji MCP musi być obiektem.")

    servers: dict[str, MCPServerConfig] = {}

    for name, server_raw in servers_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise MCPConfigError("Każdy serwer MCP musi mieć niepustą nazwę.")
        if not isinstance(server_raw, dict):
            raise MCPConfigError(f"MCP '{name}': konfiguracja musi być obiektem.")

        transport = str(server_raw.get("transport", "stdio")).strip().lower()
        if transport != "stdio":
            Logger.printWarning(
                f"MCP '{name}': transport '{transport}' nie jest obsługiwany "
                "(na razie tylko stdio); pomijam serwer."
            )
            continue

        command_raw = server_raw.get("command")
        if not isinstance(command_raw, str) or not command_raw.strip():
            raise MCPConfigError(f"MCP '{name}': pole 'command' jest wymagane.")

        args = _as_string_list(server_raw.get("args", []), "args", name)
        args = [_expand(arg) for arg in args]
        command = _expand(command_raw.strip())

        env_raw = server_raw.get("env", {})
        if env_raw is None:
            env_raw = {}
        if not isinstance(env_raw, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env_raw.items()
        ):
            raise MCPConfigError(
                f"MCP '{name}': pole 'env' musi być obiektem napisów."
            )
        env = {key: _expand(value) for key, value in env_raw.items()} or None

        cwd_raw = server_raw.get("cwd")
        cwd: str | None
        if cwd_raw is None:
            cwd = None
        elif isinstance(cwd_raw, str):
            cwd = _expand(cwd_raw)
        else:
            raise MCPConfigError(f"MCP '{name}': pole 'cwd' musi być napisem.")

        timeout_raw = server_raw.get("timeout_s", DEFAULT_MCP_TIMEOUT)
        if isinstance(timeout_raw, bool) or not isinstance(timeout_raw, (int, float)):
            raise MCPConfigError(
                f"MCP '{name}': pole 'timeout_s' musi być liczbą."
            )
        timeout_s = float(timeout_raw)
        if timeout_s <= 0:
            raise MCPConfigError(
                f"MCP '{name}': pole 'timeout_s' musi być większe od zera."
            )

        startup_timeout_raw = server_raw.get(
            "startup_timeout_s", DEFAULT_MCP_STARTUP_TIMEOUT
        )
        if isinstance(startup_timeout_raw, bool) or not isinstance(
            startup_timeout_raw, (int, float)
        ):
            raise MCPConfigError(
                f"MCP '{name}': pole 'startup_timeout_s' musi być liczbą."
            )
        startup_timeout_s = float(startup_timeout_raw)
        if startup_timeout_s <= 0:
            raise MCPConfigError(
                f"MCP '{name}': pole 'startup_timeout_s' musi być większe od zera."
            )

        enabled_raw = server_raw.get("enabled", True)
        if not isinstance(enabled_raw, bool):
            raise MCPConfigError(f"MCP '{name}': pole 'enabled' musi być bool.")

        allow_all, allow_tools, deny_tools = _parse_tools(
            server_raw.get("tools"), name
        )

        servers[name] = MCPServerConfig(
            name=name,
            command=command,
            args=tuple(args),
            env=env,
            cwd=cwd,
            timeout_s=timeout_s,
            startup_timeout_s=startup_timeout_s,
            enabled=enabled_raw,
            allow_all_tools=allow_all,
            allow_tools=allow_tools,
            deny_tools=deny_tools,
        )

    return servers


class MCPHandle:
    """Zarządza serwerami MCP i mapuje ich narzędzia na ``ToolHandle``."""

    def __init__(
        self,
        config_path: str | Path | None,
        tool_handle: ToolHandle,
    ) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.tool_handle = tool_handle
        self._runtime = MCPRuntime()
        self._servers: dict[str, MCPServerConfig] = {}
        self._params: dict[str, StdioServerParameters] = {}
        self._tools_by_server: dict[str, list[str]] = {}
        self._exposed: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()
        self._started = False

    # ------------------------------------------------------------------
    # Start / close
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True

        try:
            servers = load_servers_config(self.config_path)
        except MCPConfigError as exc:
            Logger.printError(str(exc))
            return

        self._servers = {
            name: config for name, config in servers.items() if config.enabled
        }
        disabled = [name for name, config in servers.items() if not config.enabled]
        if disabled:
            Logger.printInfo(
                f"MCP: pomijam wyłączone serwery: {', '.join(sorted(disabled))}."
            )

        if not self._servers:
            Logger.printInfo("MCP: brak włączonych serwerów.")
            return

        self._runtime.start()

        with ThreadPoolExecutor(max_workers=min(8, len(self._servers))) as executor:
            futures = {
                executor.submit(self._connect_server, name, config): (name, config)
                for name, config in self._servers.items()
            }
            for future in as_completed(futures):
                name, config = futures[future]
                try:
                    tools = future.result()
                except Exception as exc:
                    Logger.printWarning(f"MCP '{name}' niedostępny: {exc}")
                    continue
                try:
                    self._register_server(name, config, tools)
                except Exception as exc:
                    Logger.printWarning(
                        f"MCP '{name}': nie udało się zarejestrować narzędzi: {exc}"
                    )

        if not self._tools_by_server:
            Logger.printWarning("MCP: żaden serwer nie udostępnił narzędzi.")

    def _connect_server(
        self,
        name: str,
        config: MCPServerConfig,
    ) -> list[Any]:
        params = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env=config.env,
            cwd=config.cwd,
        )
        with self._lock:
            self._params[name] = params

        init_result, tools = self._runtime.connect(
            name, params, config.startup_timeout_s
        )
        server_info = getattr(init_result, "server_info", None)
        version = getattr(server_info, "version", None)
        suffix = f" (wersja {version})" if version else ""
        Logger.printInfo(
            f"MCP '{name}': połączono{suffix}, narzędzia: {len(tools)}."
        )
        return tools

    def close(self) -> None:
        with self._lock:
            self._started = False
            self._remove_all_tools()
        self._runtime.close()

    # ------------------------------------------------------------------
    # Rejestracja narzędzi
    # ------------------------------------------------------------------
    def _is_allowed(self, config: MCPServerConfig, tool_name: str) -> bool:
        if "*" in config.deny_tools or tool_name in config.deny_tools:
            return False
        if config.allow_all_tools:
            return True
        return tool_name in config.allow_tools

    def _existing_exposed_name(self, server_name: str, remote_name: str) -> str | None:
        for exposed, pair in self._exposed.items():
            if pair == (server_name, remote_name):
                return exposed
        return None

    def _register_server(
        self,
        name: str,
        config: MCPServerConfig,
        tools: list[Any],
    ) -> None:
        allowed: list[str] = []
        skipped = 0

        with self._lock:
            for tool in tools:
                remote_name = getattr(tool, "name", None)
                if not isinstance(remote_name, str) or not remote_name:
                    skipped += 1
                    continue
                if not self._is_allowed(config, remote_name):
                    skipped += 1
                    continue

                try:
                    existing_names = set(self._exposed)
                    existing_names.update(
                        tool_schema["function"]["name"]
                        for tool_schema in self.tool_handle.get_tools_list()
                    )
                    exposed = self._existing_exposed_name(
                        name, remote_name
                    ) or exposed_tool_name(
                        name, remote_name, existing_names
                    )
                    schema = sanitize_tool_input_schema(
                        getattr(tool, "input_schema", None)
                    )
                    description = getattr(tool, "description", None) or (
                        f"Narzędzie MCP '{name}/{remote_name}'."
                    )
                    executor = self._make_executor(name, remote_name)

                    self.tool_handle.create_tool_from_schema(
                        name=exposed,
                        description=description,
                        parameters=schema,
                        function=executor,
                        overwrite=True,
                    )
                except Exception as exc:
                    Logger.printWarning(
                        f"MCP '{name}': pomijam narzędzie '{remote_name}': {exc}"
                    )
                    skipped += 1
                    continue

                self._exposed[exposed] = (name, remote_name)
                allowed.append(remote_name)

            self._tools_by_server[name] = allowed

        Logger.printInfo(
            f"MCP '{name}': zarejestrowano {len(allowed)} narzędzi"
            + (f", pominięto {skipped}." if skipped else ".")
        )

    def _make_executor(self, server_name: str, remote_name: str):
        def executor(**kwargs):
            return self.call_tool(server_name, remote_name, kwargs)

        executor.__name__ = f"mcp_{server_name}_{remote_name}"
        return executor

    def _remove_server_tools(self, server_name: str) -> None:
        with self._lock:
            for exposed, (server, _tool) in list(self._exposed.items()):
                if server == server_name:
                    self.tool_handle.delete_tool(exposed)
                    self._exposed.pop(exposed, None)
            self._tools_by_server.pop(server_name, None)

    def _remove_all_tools(self) -> None:
        with self._lock:
            for exposed in list(self._exposed):
                self.tool_handle.delete_tool(exposed)
            self._exposed.clear()
            self._tools_by_server.clear()

    # ------------------------------------------------------------------
    # Wywołanie narzędzia
    # ------------------------------------------------------------------
    def call_tool(
        self,
        server_name: str,
        remote_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        config = self._servers.get(server_name)
        if config is None:
            raise MCPConnectionError(
                f"Nieznany serwer MCP '{server_name}'."
            )
        if not self._is_allowed(config, remote_name):
            raise MCPToolError(
                f"Narzędzie MCP '{server_name}/{remote_name}' jest zablokowane "
                "przez politykę."
            )

        for attempt in range(2):
            try:
                result = self._runtime.call_tool(
                    server_name,
                    remote_name,
                    arguments,
                    config.timeout_s,
                )
                return self._normalize_result(server_name, remote_name, result)
            except MCPConnectionError as exc:
                if attempt == 1:
                    raise
                Logger.printWarning(
                    f"MCP '{server_name}' zerwał połączenie ({exc}); "
                    "próbuję połączyć ponownie."
                )
                self._reconnect(server_name, config)

        raise MCPConnectionError(
            f"Nie udało się wywołać narzędzia MCP '{server_name}/{remote_name}'."
        )

    def _reconnect(self, name: str, config: MCPServerConfig) -> None:
        with self._lock:
            params = self._params.get(name)
            if params is None:
                raise MCPConnectionError(
                    f"Brak parametrów połączenia dla serwera MCP '{name}'."
                )
            _init_result, tools = self._runtime.restart(
                name, params, config.startup_timeout_s
            )
            self._remove_server_tools(name)
            self._register_server(name, config, tools)

    @staticmethod
    def _normalize_result(
        server_name: str,
        remote_name: str,
        result: Any,
    ) -> dict[str, Any]:
        if getattr(result, "result_type", "complete") == "input_required":
            raise MCPToolError(
                f"Narzędzie MCP '{server_name}/{remote_name}' wymaga dodatkowego "
                "wejścia od klienta, co nie jest jeszcze obsługiwane."
            )

        text_parts: list[str] = []
        attachments: list[dict[str, Any]] = []

        for block in getattr(result, "content", None) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    text_parts.append(text)
            elif block_type in ("image", "audio"):
                attachments.append(
                    {
                        "type": block_type,
                        "mime_type": getattr(block, "mime_type", None),
                        "data_chars": len(getattr(block, "data", "") or ""),
                    }
                )
            elif block_type == "resource_link":
                attachments.append(
                    {
                        "type": "resource_link",
                        "name": getattr(block, "name", None),
                        "uri": getattr(block, "uri", None),
                        "mime_type": getattr(block, "mime_type", None),
                        "size": getattr(block, "size", None),
                    }
                )
            elif block_type == "resource":
                resource = getattr(block, "resource", None)
                text = getattr(resource, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
                else:
                    attachments.append(
                        {
                            "type": "resource",
                            "uri": getattr(resource, "uri", None),
                            "mime_type": getattr(resource, "mime_type", None),
                            "blob_chars": len(getattr(resource, "blob", "") or ""),
                        }
                    )

        payload: dict[str, Any] = {}
        if text_parts:
            payload["text"] = "\n".join(text_parts)

        structured = getattr(result, "structured_content", None)
        if structured is not None:
            payload["structured"] = structured

        if attachments:
            payload["attachments"] = attachments

        if not payload:
            payload["text"] = ""

        if getattr(result, "is_error", False):
            message = payload.get("text") or (
                f"Narzędzie MCP '{server_name}/{remote_name}' zwróciło błąd."
            )
            raise MCPToolError(message)

        return payload

    # ------------------------------------------------------------------
    # Diagnostyka
    # ------------------------------------------------------------------
    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": name,
                    "enabled": config.enabled,
                    "connected": self._runtime.is_alive(name),
                    "tools": list(self._tools_by_server.get(name, [])),
                }
                for name, config in self._servers.items()
            ]
