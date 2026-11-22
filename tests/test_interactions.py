import time

from core.Interactions import (
    POLL_CANCELLED,
    POLL_TIMEOUT,
    InteractionBroker,
    public_approval_event,
    public_question_event,
)


def test_approval_once_is_consumed_once():
    broker = InteractionBroker(timeout=5)
    pending = broker.register_approval(
        tool="delete_file",
        risk=("write", "destructive"),
        arguments={"path": "a.txt"},
        chat_id="chat",
        run_id="run",
    )
    assert public_approval_event(pending)["risk"] == ["write", "destructive"]

    assert broker.resolve_approval(pending.id, "allow_once").ok
    assert broker.poll(pending) == "allow_once"
    assert not broker.resolve_approval(pending.id, "allow_once").ok


def test_session_allow_and_forget_chat():
    broker = InteractionBroker()
    pending = broker.register_approval(
        tool="write_file",
        risk=("write",),
        arguments={},
        chat_id="chat",
        run_id="run",
    )
    assert broker.resolve_approval(pending.id, "allow_session").ok
    assert broker.poll(pending) == "allow_session"
    assert broker.session_allowed("chat", "write_file")
    assert not broker.session_allowed("other", "write_file")
    broker.forget_chat("chat")
    assert not broker.session_allowed("chat", "write_file")


def test_timeout_and_cancel_run():
    broker = InteractionBroker(timeout=0.01)
    pending = broker.register_approval(
        tool="run_python",
        risk=("destructive",),
        arguments={},
        chat_id="chat",
        run_id="run",
    )
    time.sleep(0.03)
    assert broker.poll(pending) == POLL_TIMEOUT

    pending = broker.register_approval(
        tool="run_python",
        risk=("destructive",),
        arguments={},
        chat_id="chat",
        run_id="run-2",
    )
    broker.cancel_run("run-2")
    assert broker.poll(pending) == POLL_CANCELLED


def test_question_answer_and_event():
    broker = InteractionBroker()
    pending = broker.register_question(
        question="Który wariant?",
        options=["A", "B"],
        chat_id="chat",
        run_id="run",
    )
    event = public_question_event(pending)
    assert event["type"] == "user_input_required"
    assert event["options"] == ["A", "B"]
    assert broker.resolve_question(pending.id, {"answer": "B", "option": "B"}).ok
    assert broker.poll(pending) == {"answer": "B", "option": "B"}
