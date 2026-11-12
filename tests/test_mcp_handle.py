from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handlers.MCPHandle import MCPConfigError, MCPHandle, load_servers_config
from handlers.MCPSchema import (
    exposed_tool_name,
    sanitize_schema,
    sanitize_tool_input_schema,
)
from handlers.ToolHandle import ToolHandle


ECHO_SERVER = PROJECT_ROOT / "tests" / "mcp" / "mcp_echo_server.py"


def _config(allow: list[str], timeout_s: float = 8.0, enabled: bool = True) -> dict:
    return {
        "servers": {
            "echo": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(ECHO_SERVER)],
                "cwd": str(PROJECT_ROOT),
                "timeout_s": timeout_s,
                "enabled": enabled,
                "tools": {"allow": allow},
            }
        }
    }


class MCPSchemaTests(unittest.TestCase):
    def test_namespacing_sanitizes_names(self):
        self.assertEqual(
            exposed_tool_name("my server", "read file"),
            "mcp_my_server_read_file",
        )

    def test_name_collision_gets_hash_suffix(self):
        first = exposed_tool_name("a", "b_c")
        second = exposed_tool_name("a_b", "c", {first})

        self.assertNotEqual(first, second)
        self.assertTrue(second.startswith(first))

    def test_name_length_is_limited(self):
        name = exposed_tool_name("x" * 100, "y" * 100)
        self.assertLessEqual(len(name), 64)

    def test_sanitize_schema_preserves_common_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ścieżka.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["read", "write"],
                },
            },
            "required": ["path", "missing"],
            "additionalProperties": False,
        }

        result = sanitize_schema(schema)

        self.assertEqual(result["type"], "object")
        self.assertEqual(result["properties"]["path"]["type"], "string")
        self.assertEqual(result["properties"]["mode"]["enum"], ["read", "write"])
        self.assertFalse(result["additionalProperties"])
        self.assertEqual(result["required"], ["path"])

    def test_sanitize_schema_resolves_local_ref(self):
        schema = {
            "type": "object",
            "properties": {"name": {"$ref": "#/$defs/Name"}},
            "required": ["name"],
            "$defs": {
                "Name": {
                    "type": "string",
                    "description": "Nazwa.",
                }
            },
        }

        result = sanitize_schema(schema)

        self.assertEqual(result["properties"]["name"]["type"], "string")
        self.assertEqual(result["properties"]["name"]["description"], "Nazwa.")

    def test_sanitize_schema_infers_type_from_enum(self):
        schema = {
            "type": "object",
            "properties": {"mode": {"enum": ["read", "write"]}},
        }

        result = sanitize_schema(schema)

        self.assertEqual(result["properties"]["mode"]["type"], "string")
        self.assertEqual(result["properties"]["mode"]["enum"], ["read", "write"])

    def test_non_object_schema_is_wrapped_as_empty_object(self):
        result = sanitize_tool_input_schema({"type": "string"})
        self.assertEqual(result, {"type": "object", "properties": {}, "required": []})


class MCPConfigTests(unittest.TestCase):
    def _write(self, payload) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        path = directory / "mcp_servers.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_file_returns_empty_config(self):
        self.assertEqual(load_servers_config("/nonexistent/mcp_servers.json"), {})

    def test_invalid_json_raises_config_error(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        path = directory / "mcp_servers.json"
        path.write_text("{invalid", encoding="utf-8")

        with self.assertRaises(MCPConfigError):
            load_servers_config(path)

    def test_policy_parsing(self):
        path = self._write(
            {
                "servers": {
                    "echo": {
                        "command": sys.executable,
                        "args": [str(ECHO_SERVER)],
                        "tools": {
                            "allow": ["echo", "add"],
                            "deny": ["add"],
                            "ask": ["slow"],
                        },
                    }
                }
            }
        )

        servers = load_servers_config(path)
        config = servers["echo"]

        self.assertFalse(config.allow_all_tools)
        self.assertEqual(config.allow_tools, frozenset({"echo", "add"}))
        self.assertEqual(config.deny_tools, frozenset({"add", "slow"}))

    def test_allow_star_means_all(self):
        path = self._write(
            {
                "servers": {
                    "echo": {
                        "command": sys.executable,
                        "args": [str(ECHO_SERVER)],
                        "tools": {"allow": "*"},
                    }
                }
            }
        )

        self.assertTrue(load_servers_config(path)["echo"].allow_all_tools)


class MCPHandleIntegrationTests(unittest.TestCase):
    def _make_handle(
        self,
        allow: list[str],
        timeout_s: float = 8.0,
        enabled: bool = True,
    ) -> tuple[MCPHandle, ToolHandle]:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        config_path = directory / "mcp_servers.json"
        config_path.write_text(
            json.dumps(_config(allow, timeout_s=timeout_s, enabled=enabled)),
            encoding="utf-8",
        )

        tool_handle = ToolHandle()
        handle = MCPHandle(config_path, tool_handle)
        handle.start()
        self.addCleanup(handle.close)
        return handle, tool_handle

    def test_registers_only_allowed_tools(self):
        handle, tool_handle = self._make_handle(["echo", "add"])

        exposed = {
            tool["function"]["name"] for tool in tool_handle.get_tools_list()
        }
        self.assertIn("mcp_echo_echo", exposed)
        self.assertIn("mcp_echo_add", exposed)
        self.assertNotIn("mcp_echo_slow", exposed)

        result = json.loads(
            tool_handle.execute_tool("mcp_echo_slow", {"seconds": 0.1})
        )
        self.assertFalse(result["success"])

        status = handle.status()
        self.assertEqual(status[0]["name"], "echo")
        self.assertTrue(status[0]["connected"])

    def test_calls_echo_and_add(self):
        _handle, tool_handle = self._make_handle(["echo", "add"])

        echo_result = json.loads(
            tool_handle.execute_tool("mcp_echo_echo", {"text": "hello"})
        )
        self.assertTrue(echo_result["success"])
        self.assertEqual(echo_result["result"]["text"], "hello")

        add_result = json.loads(
            tool_handle.execute_tool("mcp_echo_add", {"a": 2, "b": 3})
        )
        self.assertTrue(add_result["success"])
        self.assertEqual(add_result["result"]["structured"]["result"], 5)

    def test_timeout_returns_error_and_session_stays_alive(self):
        _handle, tool_handle = self._make_handle(["echo", "slow"], timeout_s=0.5)

        result = json.loads(
            tool_handle.execute_tool("mcp_echo_slow", {"seconds": 2})
        )
        self.assertFalse(result["success"])
        self.assertIn("limit czasu", result["error"])

        echo_result = json.loads(
            tool_handle.execute_tool("mcp_echo_echo", {"text": "still alive"})
        )
        self.assertTrue(echo_result["success"])
        self.assertEqual(echo_result["result"]["text"], "still alive")

    def test_tool_error_returns_structured_failure(self):
        _handle, tool_handle = self._make_handle(["echo", "fail"])

        result = json.loads(
            tool_handle.execute_tool("mcp_echo_fail", {"message": "boom"})
        )
        self.assertFalse(result["success"])
        self.assertIn("fail", result["error"])

        echo_result = json.loads(
            tool_handle.execute_tool("mcp_echo_echo", {"text": "after error"})
        )
        self.assertTrue(echo_result["success"])

    def test_crash_is_reported_and_next_call_reconnects(self):
        _handle, tool_handle = self._make_handle(["echo", "crash"], timeout_s=5.0)

        crash_result = json.loads(
            tool_handle.execute_tool("mcp_echo_crash", {})
        )
        self.assertFalse(crash_result["success"])

        time.sleep(0.2)
        echo_result = json.loads(
            tool_handle.execute_tool("mcp_echo_echo", {"text": "reconnected"})
        )
        self.assertTrue(echo_result["success"])
        self.assertEqual(echo_result["result"]["text"], "reconnected")

    def test_close_removes_tools_and_stops_runtime(self):
        handle, tool_handle = self._make_handle(["echo"])

        self.assertTrue(tool_handle.get_tools_list())

        handle.close()

        self.assertEqual(tool_handle.get_tools_list(), [])
        self.assertFalse(handle._runtime._thread.is_alive())

    def test_disabled_server_is_not_started(self):
        handle, tool_handle = self._make_handle(["echo"], enabled=False)

        self.assertEqual(tool_handle.get_tools_list(), [])
        self.assertEqual(handle.status(), [])
        self.assertFalse(handle._runtime._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
