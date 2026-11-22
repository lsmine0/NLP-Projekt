from types import SimpleNamespace

import core.main as app_main
from core.Interactions import InteractionBroker


def test_cli_approval_once_session_and_deny(monkeypatch, capsys):
    broker = InteractionBroker()
    runtime = SimpleNamespace(interaction_broker=broker)
    event = {
        "type": "approval_required",
        "id": "apr_test",
        "tool": "delete_file",
        "source": "builtin",
        "risk": ["write", "destructive"],
        "arguments": {"path": "a.txt"},
    }

    pending = broker.register_approval(
        tool="delete_file", risk=("write", "destructive"),
        arguments={"path": "a.txt"}, chat_id="chat", run_id="run",
    )
    event["id"] = pending.id
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    assert app_main._cli_approval(runtime, event) == "allow_once"
    assert broker.poll(pending) == "allow_once"

    pending = broker.register_approval(
        tool="write_file", risk=("write",), arguments={}, chat_id="chat", run_id="run2",
    )
    event["id"] = pending.id
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    assert app_main._cli_approval(runtime, event) == "allow_session"
    assert broker.poll(pending) == "allow_session"
    assert broker.session_allowed("chat", "write_file")

    pending = broker.register_approval(
        tool="delete_file", risk=("write", "destructive"), arguments={},
        chat_id="chat", run_id="run3",
    )
    event["id"] = pending.id
    monkeypatch.setattr("builtins.input", lambda prompt="": "3")
    assert app_main._cli_approval(runtime, event) == "deny"
    assert broker.poll(pending) == "deny"
    captured = capsys.readouterr().out
    assert "delete_file" in captured


def test_cli_question_option_and_custom(monkeypatch):
    broker = InteractionBroker()
    runtime = SimpleNamespace(interaction_broker=broker)
    event = {
        "type": "user_input_required",
        "id": "",
        "question": "Wariant?",
        "options": ["A", "B"],
    }

    pending = broker.register_question(
        question="Wariant?", options=["A", "B"], chat_id="chat", run_id="run",
    )
    event["id"] = pending.id
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    app_main._cli_question(runtime, event)
    assert broker.poll(pending) == {"answer": "B", "option": "B"}

    pending = broker.register_question(
        question="Wariant?", options=["A", "B"], chat_id="chat", run_id="run2",
    )
    event["id"] = pending.id
    monkeypatch.setattr("builtins.input", lambda prompt="": "coś innego")
    app_main._cli_question(runtime, event)
    assert broker.poll(pending) == {"answer": "coś innego"}

    pending = broker.register_question(
        question="Wariant?", options=["A", "B"], chat_id="chat", run_id="run3",
    )
    event["id"] = pending.id
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    app_main._cli_question(runtime, event)
    assert broker.poll(pending) == {"cancelled": True}
