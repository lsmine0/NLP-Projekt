import json
from types import SimpleNamespace as NS

from core.AgentFlow import AgentFlow
from core.ChatStore import DEFAULT_SYSTEM_PROMPT
from core.PromptBuilder import build_system_prompt, build_tool_prompt
from handlers.OllamaHandle import OllamaHandler
from handlers.ToolHandle import ToolHandle
from tools.ToolBuilder import build_tools


class ContextProvider:
    def __init__(self, tools, num_ctx=4096):
        self.ToolsHandle = tools
        self.model = "test"
        self.base_url = "http://fake/v1"
        self.num_ctx = num_ctx

    def GetNumCtx(self):
        return self.num_ctx

    def EnsureContextBudget(self, *args, **kwargs):
        return None

    def Chat(self, *args, **kwargs):
        return NS(choices=[NS(message=NS(content="ok", tool_calls=None))])

    def ChatStream(self, *args, **kwargs):
        yield NS(choices=[NS(delta=NS(content="ok", tool_calls=None, reasoning=None, reasoning_content=None))])

    def close(self):
        pass


def test_small_context_reserve_does_not_exceed_window():
    tools = ToolHandle()
    agent = AgentFlow(provider=ContextProvider(tools, 4096), tool_handle=tools)
    agent.RefreshModelContext()
    assert agent.num_ctx == 4096
    assert 256 <= agent.reserve_tokens <= 1024
    assert agent.max_context_tokens >= 2048


def test_compact_tool_schemas_fit_small_context():
    tools = build_tools(workspace_dir=".")
    handler = OllamaHandler(tool_handle=tools)
    full = tools.get_tools_list()
    compact = handler._CompactTools(full)

    full_tokens = len(json.dumps(full, ensure_ascii=False)) // 4
    compact_tokens = len(json.dumps(compact, ensure_ascii=False)) // 4
    assert compact_tokens < full_tokens
    assert compact_tokens < 2500
    assert all(len(item["function"]["description"]) <= 81 for item in compact)
    for item in compact:
        for schema in item["function"]["parameters"].get("properties", {}).values():
            assert "description" not in schema


def test_context_budget_keeps_user_message_with_compact_prompt():
    tools = build_tools(workspace_dir=".")
    handler = OllamaHandler(tool_handle=tools)
    handler.model = "test"
    handler.GetNumCtx = lambda: 4096
    tool_prompt = build_tool_prompt(tools, {})
    system_prompt = build_system_prompt(DEFAULT_SYSTEM_PROMPT, tool_prompt, {})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Jaka jest aktualna cena dolara?"},
    ]

    handler.EnsureContextBudget(messages, summarizer=None, reserve_tokens=1024)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "Jaka jest aktualna cena dolara?"
    assert handler._EstimateToolTokens() < 2500
