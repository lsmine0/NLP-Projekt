from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from types import SimpleNamespace

# Make project root importable so this works from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.AgentFlow as af_mod
AgentFlow = af_mod.AgentFlow
import handlers.OllamaHandle as oh_mod
from handlers.OllamaHandle import OllamaHandler
from handlers.ContextHandle import ContextHandle
from core import Logger

DEFAULT_TOOL_CHARS = af_mod.DEFAULT_TOOL_CHARS
DEFAULT_TOOL_TIMEOUT = af_mod.DEFAULT_TOOL_TIMEOUT
DEFAULT_RESERVE_TOKENS = af_mod.DEFAULT_RESERVE_TOKENS
DEFAULT_NUM_CTX = oh_mod.DEFAULT_NUM_CTX


# ============================================================
#  Helpers
# ============================================================

def _fill(length):
    return "x" * length


def _make_history(system="sys", pairs=15, filler=500):
    """system + N user/assistant pairs of `filler`-char messages."""
    msgs = [ContextHandle.Create(system, role="system")]
    for _ in range(pairs):
        msgs.append(ContextHandle.Create(_fill(filler), role="user"))
        msgs.append(ContextHandle.Create(_fill(filler), role="assistant"))
    return msgs


# --- fake message / response builders for AgentFlow tests ---

class _Msg:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None, role="assistant"):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.role = role

    def model_dump(self, exclude_none=False):
        d = {
            "role": self.role,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": self.tool_calls,
        }
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


def _func(name, args):
    return SimpleNamespace(name=name, arguments=args)


def _response(content=None, tool_calls=None, reasoning_content=None):
    """Non-streaming fake response: .choices[0].message."""
    tcs = [
        SimpleNamespace(id=f"c{i}", function=_func(name, args))
        for i, (name, args) in enumerate(tool_calls or [])
    ]
    msg = _Msg(content=content, reasoning_content=reasoning_content, tool_calls=tcs or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _delta(content=None, tool_calls=None):
    """Streaming fake chunk: .choices[0].delta."""
    tcs = [
        SimpleNamespace(id=f"c{i}", function=_func(name, args))
        for i, (name, args) in enumerate(tool_calls or [])
    ]
    delta = SimpleNamespace(content=content, tool_calls=tcs or None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


# --- fake OllamaHandler for EnsureContextBudget tests ---

class _FakeOllama:
    def __init__(self, num_ctx, reserve):
        self.model = "test-model"
        self.num_ctx = num_ctx
        self.reserve_tokens = reserve
        self.auto_budget = True

    def GetNumCtx(self):
        return self.num_ctx


# --- fake OllamaHandler for AgentFlow tests ---

class _FakeToolsHandle:
    def __init__(self):
        self.execute_tool = None  # set per test

    def register_function(self, name, fn):
        pass

    def create_tool(self, *a, **k):
        return {}


class _FakeAgentOllama:
    def __init__(self, num_ctx=4096, reserve=8000):
        self.model = "test-model"
        self.num_ctx = num_ctx
        self.reserve_tokens = reserve
        self.auto_budget = True
        self._responses = []   # non-streaming Chat results
        self._streams = []     # streaming ChatStream results
        self.tools = _FakeToolsHandle()

    def GetNumCtx(self):
        return self.num_ctx

    def EnsureContextBudget(self, *a, **k):
        pass  # no-op: test the agent flow, not the budgeting

    def Chat(self, messages, **kwargs):
        return self._responses.pop(0)

    def ChatStream(self, messages, **kwargs):
        return iter(self._streams.pop(0))


def _make_agent(num_ctx=4096, reserve=8000, max_tool_calls=15, system_prompt="sys"):
    agent = AgentFlow(
        base_url="http://localhost:11434/v1",
        key="ollam",
        system_prompt=system_prompt,
        max_tool_calls=max_tool_calls,
    )
    fake = _FakeAgentOllama(num_ctx=num_ctx, reserve=reserve)
    agent.ollama = fake
    agent.tools = fake.tools
    return agent


# ============================================================
#  ContextHandle tests (pure, no mocks)
# ============================================================

def test_context_handle():
    print("  - ContextHandle")
    history = [
        ContextHandle.Create("You are a helpful assistant.", role="system"),
        ContextHandle.Create("Hello, can you check my balance?", role="user"),
        ContextHandle.Create("Checking balance...", role="assistant", tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_balance", "arguments": '{"account": "12345"}'}
            }
        ]),
        ContextHandle.Create('{"balance": 1500.50, "currency": "USD"}', role="tool", tool_call_id="call_123"),
        ContextHandle.Create("Your balance is 1500.50 USD.", role="assistant"),
        ContextHandle.Create("Thanks! Now transfer 100 USD.", role="user"),
    ]

    # tokens
    assert ContextHandle.CountTokens("") == 0, "empty text -> 0 tokens"
    assert ContextHandle.CountTokens("12345678") == 2, "CountTokens failed"
    assert ContextHandle.MessageTokens(ContextHandle.Create("hello", role="user")) == 5 + ContextHandle.CountTokens("hello"), "MessageTokens overhead wrong"
    assert ContextHandle.HistoryTokens([ContextHandle.Create("a", role="user")]) == ContextHandle.MessageTokens(ContextHandle.Create("a", role="user")), "HistoryTokens wrong"

    # Trim: keep system + tail, drop middle
    trimmed, removed, over = ContextHandle.Trim(history, max_tokens=45)
    assert trimmed[0]["role"] == "system", "Trim dropped system prompt"
    assert trimmed[-1]["role"] == "user", "Trim dropped tail user message"
    assert removed > 0, "Trim removed nothing"
    assert not over, "Trimmed history over budget"

    # TrimHard
    res, removed2 = ContextHandle.TrimHard(history, max_tokens=45)
    assert res[0]["role"] == "system", "TrimHard dropped system prompt"
    assert removed2 >= 0, "TrimHard removed negative"

    # Summarize: LLM summary -> system-role message, original prompt kept, fits
    cand, fits = ContextHandle.Summarize(history, max_tokens=60, summarize_fn=lambda old: "S.")
    assert fits, "summary candidate did not fit"
    assert cand[0]["role"] == "system", "original system prompt lost"
    assert cand[0]["content"] == "You are a helpful assistant.", "original prompt content changed"
    assert any(m["role"] == "system" and m["content"].startswith("Stary fragment historii") for m in cand), "summary not inserted as system role"
    assert cand[-1]["role"] == "user", "tail user message lost after summarizing"

    # Summarize: LLM returns None -> Trim fallback
    ref, _, _ = ContextHandle.Trim(history, max_tokens=60)
    cand, fits = ContextHandle.Summarize(history, max_tokens=60, summarize_fn=lambda old: None)
    assert cand == ref, "None summarizer did not fall back to Trim"
    assert fits, "Trim fallback over budget"

    # Summarize: LLM raises -> Trim fallback
    def boom(old):
        raise ValueError("boom")
    cand, fits = ContextHandle.Summarize(history, max_tokens=60, summarize_fn=boom)
    assert cand == ref, "raising summarizer did not fall back to Trim"

    # Summarize: summary too big -> Trim fallback
    cand, fits = ContextHandle.Summarize(history, max_tokens=60, summarize_fn=lambda old: "A" * 10000)
    assert cand == ref, "oversized summary did not fall back to Trim"

    # TruncateJson: large dict -> truncated flag, valid JSON
    large = {"status": "ok", "data": "A" * 2000}
    out, flag = ContextHandle.TruncateJson(json.dumps(large), max_tokens=50)
    assert flag is True, "large dict not truncated"
    assert json.loads(out).get("truncated") is True, "truncated flag missing"
    assert len(out) < len(json.dumps(large)), "truncated string not shorter"

    # TruncateJson: small dict passes through unchanged
    small = {"a": 1}
    out, flag = ContextHandle.TruncateJson(json.dumps(small), max_tokens=50)
    assert flag is False, "small dict wrongly truncated"
    assert out == json.dumps(small), "small dict changed"

    # TruncateJson: empty + non-dict
    out, flag = ContextHandle.TruncateJson("", max_tokens=50)
    assert out == "" and flag is False, "empty payload wrong"
    out, flag = ContextHandle.TruncateJson(json.dumps(["x" * 2000]), max_tokens=50)
    assert flag is True and out.endswith(" [TRUNCATED]"), "non-dict truncation wrong"

    print("  OK: ContextHandle")


# ============================================================
#  OllamaHandler.EnsureContextBudget tests (mocked, no network)
# ============================================================

def test_ensure_context_budget():
    print("  - OllamaHandler.EnsureContextBudget")
    # budget = max(1024, int((num_ctx - reserve - 128) / ratio))
    # num_ctx=4096, reserve=256, ratio=1.0 -> budget = 3712
    # ~3906-token history is over budget
    fake = _FakeOllama(num_ctx=4096, reserve=256)
    hist = _make_history(pairs=15, filler=500)
    assert ContextHandle.HistoryTokens(hist) > 3712, "test history not over budget"

    # 1. good (short) summary -> fits, system prompt kept
    out = list(hist)
    OllamaHandler.EnsureContextBudget(fake, out, summarizer=lambda old: "ok summary", ratio=1.0, reserve_tokens=256)
    assert out[0]["content"] == "sys", "system prompt lost after summary"
    assert any(m["role"] == "system" and m["content"].startswith("Stary fragment historii") for m in out), "summary message missing"
    assert any(m["role"] == "user" for m in out), "tail user message lost"
    assert out[-1]["role"] == "assistant", "tail assistant message lost"
    assert len(out) < len(hist), "summary did not shrink context"

    # 2. summarizer returns None -> Trim fallback
    out = list(hist)
    OllamaHandler.EnsureContextBudget(fake, out, summarizer=lambda old: None, ratio=1.0, reserve_tokens=256)
    assert out[0]["content"] == "sys", "system prompt lost (None summary)"
    assert any(m["role"] == "user" for m in out), "tail user message lost (None summary)"
    assert out[-1]["role"] == "assistant", "tail assistant message lost (None summary)"
    assert len(out) < len(hist), "None summary did not shrink context"

    # 3. summarizer raises -> Trim fallback
    out = list(hist)
    def boom(old):
        raise ValueError("boom")
    OllamaHandler.EnsureContextBudget(fake, out, summarizer=boom, ratio=1.0, reserve_tokens=256)
    assert out[0]["content"] == "sys", "system prompt lost (raising summary)"
    assert any(m["role"] == "user" for m in out), "tail user message lost (raising summary)"
    assert out[-1]["role"] == "assistant", "tail assistant message lost (raising summary)"
    assert len(out) < len(hist), "raising summary did not shrink context"

    # 4. summary too big -> Trim fallback
    out = list(hist)
    OllamaHandler.EnsureContextBudget(fake, out, summarizer=lambda old: "A" * 20000, ratio=1.0, reserve_tokens=256)
    assert out[0]["content"] == "sys", "system prompt lost (big summary)"
    assert any(m["role"] == "user" for m in out), "tail user message lost (big summary)"
    assert out[-1]["role"] == "assistant", "tail assistant message lost (big summary)"
    assert len(out) < len(hist), "big summary did not shrink context"

    # 5. no summarizer -> Trim fallback
    out = list(hist)
    OllamaHandler.EnsureContextBudget(fake, out, summarizer=None, ratio=1.0, reserve_tokens=256)
    assert out[0]["content"] == "sys", "system prompt lost (no summarizer)"
    assert any(m["role"] == "user" for m in out), "tail user message lost (no summarizer)"
    assert out[-1]["role"] == "assistant", "tail assistant message lost (no summarizer)"
    assert len(out) < len(hist), "no summarizer did not shrink context"

    # 6. in budget -> unchanged
    small = _make_history(pairs=3, filler=50)
    assert ContextHandle.HistoryTokens(small) <= 3712, "small history unexpectedly over budget"
    OllamaHandler.EnsureContextBudget(fake, small, summarizer=boom, ratio=1.0, reserve_tokens=256)
    assert len(small) == 7, "in-budget history was modified"

    print("  OK: EnsureContextBudget")


# ============================================================
#  AgentFlow.Run tests (mocked, no network)
# ============================================================

def test_agentflow_run():
    print("  - AgentFlow.Run")

    # 1. simple answer (no tool calls)
    agent = _make_agent()
    agent.ollama._responses = [_response(content="42")]
    result = agent.Run("2+2")
    assert result == "42", f"simple answer wrong: {result!r}"
    assert len(agent.context) == 3, f"context length wrong: {agent.context}"
    assert agent.context[0]["role"] == "system", "system message lost"
    assert agent.context[1]["role"] == "user", "user message lost"
    assert agent.context[2]["role"] == "assistant" and agent.context[2]["content"] == "42", "assistant answer wrong"

    # 2. tool call then answer
    agent = _make_agent()
    agent.ollama.tools.execute_tool = lambda name, args: json.dumps({"success": True, "result": 3})
    agent.ollama._responses = [
        _response(tool_calls=[("add_numbers", '{"a":1,"b":2}')]),
        _response(content="3"),
    ]
    result = agent.Run("1+2")
    assert result == "3", f"tool answer wrong: {result!r}"
    assert len(agent.context) == 5, f"tool context length wrong: {agent.context}"
    assert agent.context[2].get("tool_calls"), "assistant tool_calls missing"
    assert agent.context[3]["role"] == "tool", "tool result message lost"
    assert agent.context[4]["role"] == "assistant" and agent.context[4]["content"] == "3", "final answer wrong"

    # 3. tool-call limit -> return last answer (empty)
    agent = _make_agent(max_tool_calls=1)
    agent.ollama.tools.execute_tool = lambda name, args: json.dumps({"success": True, "result": 1})
    agent.ollama._responses = [
        _response(tool_calls=[("t", "{}")]),
        _response(tool_calls=[("t", "{}")]),
    ]
    result = agent.Run("t")
    assert result == "", "tool limit did not return empty last answer"

    # 4. tool result truncated when too big
    agent = _make_agent()
    big = json.dumps({"success": True, "result": "y" * 5000})
    agent.ollama.tools.execute_tool = lambda name, args: big
    agent.ollama._responses = [_response(tool_calls=[("t", "{}")]), _response(content="done")]
    agent.Run("t")
    tool_msg = agent.context[3]
    assert tool_msg["role"] == "tool", "tool message lost"
    assert len(tool_msg["content"]) < len(big), "truncation did not shorten result"
    assert "truncated" in tool_msg["content"].lower(), "truncation indicator missing"
    assert len(tool_msg["content"]) <= DEFAULT_TOOL_CHARS + 20, "truncated tool result too long"

    # 5. tool timeout -> timeout JSON result, agent continues
    agent = _make_agent()
    orig_timeout = af_mod.DEFAULT_TOOL_TIMEOUT
    af_mod.DEFAULT_TOOL_TIMEOUT = 0.1
    try:
        def slow_tool(name, args):
            time.sleep(0.5)
            return json.dumps({"success": True, "result": 1})
        agent.ollama.tools.execute_tool = slow_tool
        agent.ollama._responses = [_response(tool_calls=[("t", "{}")]), _response(content="done")]
        result = agent.Run("slow")
        assert result == "done", f"timeout answer wrong: {result!r}"
        tool_msg = agent.context[3]
        assert tool_msg["role"] == "tool", "tool message lost on timeout"
        assert "timed out" in tool_msg["content"], "timeout indicator missing"
    finally:
        af_mod.DEFAULT_TOOL_TIMEOUT = orig_timeout

    print("  OK: AgentFlow.Run")


# ============================================================
#  AgentFlow.RunStream tests (mocked, no network)
# ============================================================

def test_agentflow_run_stream():
    print("  - AgentFlow.RunStream")

    # 1. streaming simple answer -> yields tokens, appends assistant
    agent = _make_agent()
    agent.ollama._streams = [[_delta(content="he"), _delta(content="llo"), _delta(content=" world")]]
    tokens = list(agent.RunStream("hi"))
    assert "".join(tokens) == "hello world", f"streamed tokens wrong: {tokens}"
    assert len(agent.context) == 3, f"stream context length wrong: {agent.context}"
    assert agent.context[2]["role"] == "assistant" and agent.context[2]["content"] == "hello world", "streamed assistant content wrong"

    # 2. streaming with tool call -> executes tool, continues to answer
    agent = _make_agent()
    agent.ollama.tools.execute_tool = lambda name, args: json.dumps({"success": True, "result": 3})
    agent.ollama._streams = [
        [_delta(content=None, tool_calls=[("add_numbers", '{"a":1,"b":2}')])],
        [_delta(content="3")],
    ]
    tokens = list(agent.RunStream("1+2"))
    assert "".join(tokens) == "3", f"streamed tool answer wrong: {tokens}"
    assert len(agent.context) == 5, f"stream tool context length wrong: {agent.context}"
    assert agent.context[4]["role"] == "assistant" and agent.context[4]["content"] == "3", "final streamed answer wrong"

    print("  OK: AgentFlow.RunStream")


# ============================================================
#  Constants sanity
# ============================================================

def test_constants():
    print("  - constants")
    assert DEFAULT_TOOL_CHARS == 1024, "DEFAULT_TOOL_CHARS wrong"
    assert DEFAULT_TOOL_TIMEOUT == 10, "DEFAULT_TOOL_TIMEOUT wrong"
    assert DEFAULT_RESERVE_TOKENS == 8000, "DEFAULT_RESERVE_TOKENS wrong"
    assert DEFAULT_NUM_CTX == 16384, "DEFAULT_NUM_CTX wrong"
    print("  OK: constants")


# ============================================================
#  Runner
# ============================================================

def run_all_tests():
    tests = [
        test_constants,
        test_context_handle,
        test_ensure_context_budget,
        test_agentflow_run,
        test_agentflow_run_stream,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            Logger.printError(f"✗ FAILED {t.__name__}: {e}")
            return False
    Logger.printInfo("✓ All context-management tests passed.")
    return True


def run_demo():
    # --- original interactive demo (live Ollama required) ---
    def AddNumbers(a: float, b: float):
        return a + b

    Logger.printInfo("Initializing AgentFlow...")
    try:
        agent = AgentFlow(
            system_prompt="You are a helpful assistant with calculation tools.",
        )
    except Exception as e:
        Logger.printError(f"Failed to initialize AgentFlow (is Ollama running?): {e}")
        return

    ollama = agent.ollama
    try:
        models_data = ollama.GetModels()
        models_list = models_data.data if hasattr(models_data, "data") else models_data
        models = [model.id for model in models_list]
    except Exception as e:
        Logger.printError(f"Failed to retrieve models: {e}")
        return

    if not models:
        print("No models available.")
        return

    print("\nAvailable models:")
    for index, model in enumerate(models):
        print(f"[{index}] {model}")

    while True:
        selection = input("\nSelect model (number): ").strip()
        try:
            index = int(selection)
            if 0 <= index < len(models):
                break
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")

    selected_model = models[index]
    if not ollama.SetModel(selected_model):
        print("Failed to select model.")
        return
    print(f"Selected model: {selected_model}")

    agent.tools.create_tool(
        name="add_numbers",
        description="Adds two numbers together.",
        properties={
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        required=["a", "b"],
    )
    agent.tools.register_function("add_numbers", AddNumbers)

    stream_mode = input("Stream responses? (y/n) [y]: ").strip().lower() != "n"
    print(f"\nAgent ready using model: {selected_model}. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            print("Assistant: ", end="", flush=True)
            if stream_mode:
                for token in agent.RunStream(user_input):
                    print(token, end="", flush=True)
                print()
            else:
                answer = agent.Run(user_input)
                print(answer)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    ok = run_all_tests()
    if not ok:
        sys.exit(1)
    # Offer the interactive demo only when running in a terminal (avoids hanging in CI/pipes)
    if sys.stdin.isatty():
        demo = input("Run interactive demo? (y/n): ").strip().lower() == "y"
        if demo:
            run_demo()
