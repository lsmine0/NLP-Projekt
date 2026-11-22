import json
import os
from pathlib import Path

import pytest

from core.ToolPolicy import normalize_risk, safe_arguments
from handlers.ToolHandle import ToolHandle


def test_risk_labels_are_combinable_and_validated():
    assert normalize_risk(["write", "destructive"]) == frozenset({"write", "destructive"})
    assert normalize_risk("external,read") == frozenset({"read", "external"})
    assert normalize_risk(None) == frozenset()
    with pytest.raises(ValueError):
        normalize_risk("root")


def test_policy_resolution_exact_override_and_risk_defaults():
    tools = ToolHandle()
    tools.create_tool(
        "probe",
        "probe",
        function=lambda: {"success": True},
        risk=("write", "destructive"),
    )

    assert tools.get_policy("probe") == "ask"
    tools.set_policy_config(policy_overrides={"probe": "allow"})
    assert tools.get_policy("probe") == "allow"
    tools.set_policy_config(policy_overrides={"probe": "allow"}, risk_policies={"destructive": "deny"})
    assert tools.get_policy("probe") == "allow"

    tools.set_policy_config(policy_overrides={}, risk_policies={"destructive": "deny"})
    assert tools.get_policy("probe") == "deny"


def test_public_metadata_exposes_risk_and_policy():
    tools = ToolHandle()
    tools.create_tool(
        "delete_probe",
        "delete probe",
        function=lambda: {"success": True},
        risk=("write", "destructive"),
    )
    item = next(tool for tool in tools.public_tools() if tool["name"] == "delete_probe")
    assert item["risk"] == ["write", "destructive"]
    assert item["policy"] == "ask"


def test_file_paths_are_normalized_and_confined(tmp_path):
    allowed = tmp_path / "workspace"
    other = tmp_path / "other"
    allowed.mkdir()
    other.mkdir()

    tools = ToolHandle(allowed_directories=[allowed], primary_root=allowed)
    tools.create_tool(
        "create_probe",
        "create probe",
        properties={"path": {"type": "string"}},
        required=["path"],
        function=lambda path: {"success": True, "path": path},
        risk=("write",),
        path_args=("path",),
    )

    prepared = tools.prepare_tool_call("create_probe", {"path": "nested/../file.txt"})
    assert prepared.denial is None
    assert Path(prepared.arguments["path"]) == (allowed / "file.txt").resolve()

    escaped = tools.prepare_tool_call("create_probe", {"path": "../other/file.txt"})
    assert escaped.denial["code"] == "path_outside_allowed"

    absolute_ok = tools.prepare_tool_call("create_probe", {"path": str(other / "file.txt")})
    assert absolute_ok.denial["code"] == "path_outside_allowed"


def test_symlink_escape_is_denied(tmp_path):
    allowed = tmp_path / "workspace"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "link").symlink_to(outside, target_is_directory=True)

    tools = ToolHandle(allowed_directories=[allowed], primary_root=allowed)
    tools.create_tool(
        "read_probe",
        "read probe",
        function=lambda path: {"success": True, "path": path},
        risk=("read",),
        path_args=("path",),
    )
    prepared = tools.prepare_tool_call("read_probe", {"path": "link/secret.txt"})
    assert prepared.denial["code"] == "path_outside_allowed"


def test_protected_paths_block_writes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protected = workspace / "settings.json"
    tools = ToolHandle(
        allowed_directories=[workspace],
        primary_root=workspace,
        protected_paths=[protected],
    )
    tools.create_tool(
        "write_probe",
        "write probe",
        function=lambda path: {"success": True},
        risk=("write",),
        path_args=("path",),
    )
    prepared = tools.prepare_tool_call("write_probe", {"path": "settings.json"})
    assert prepared.denial["code"] == "protected_path"


def test_safe_arguments_redact_secrets_and_content_preview():
    class Spec:
        name = "probe"

    args = {
        "api_key": "sk-super-secret-value",
        "password": "hunter2",
        "url": "https://example.test/path?token=abc&q=ok",
        "content": "line1\nsecret=top-secret\nline3",
        "path": "/tmp/file.txt",
        "nested": {"token": "abc", "ok": "visible"},
    }
    safe = safe_arguments(Spec(), args)
    dumped = json.dumps(safe, ensure_ascii=False)
    assert "sk-super-secret-value" not in dumped
    assert "hunter2" not in dumped
    assert "top-secret" not in dumped
    assert safe["api_key"] == "***"
    assert safe["password"] == "***"
    assert safe["content"]["length"] == len(args["content"])
    assert safe["nested"]["token"] == "***"
    assert safe["nested"]["ok"] == "visible"

    url_safe = safe_arguments(Spec(), {"url": "https://user:pass@example.test/a?token=abc&q=1"})
    assert "user:pass" not in json.dumps(url_safe)
    assert "abc" not in json.dumps(url_safe)


def test_unknown_tool_arguments_are_dropped_before_policy():
    tools = ToolHandle(risk_policies={"external": "ask"})
    tools.create_tool(
        "probe",
        "probe",
        properties={"query": {"type": "string"}},
        required=["query"],
        function=lambda query, max_results=5: {"success": True, "query": query},
        risk=("external",),
    )

    prepared = tools.prepare_tool_call("probe", {"query": "kurs", "region": "pl"})
    assert prepared.denial is None
    assert prepared.requires_approval is True
    assert "region" not in prepared.arguments

    missing = tools.prepare_tool_call("probe", {"region": "pl"})
    assert missing.denial["code"] == "invalid_arguments"
