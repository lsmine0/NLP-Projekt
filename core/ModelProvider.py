"""Small structural contract matching the methods consumed by AgentFlow."""
from typing import Any, Protocol
from core.RuntimeConfig import RuntimeConfig
from handlers.ToolHandle import ToolHandle


class ModelProvider(Protocol):
    model: str | None
    base_url: str
    ToolsHandle: ToolHandle | None
    def GetModels(self) -> Any: ...
    def SetModel(self, name: str) -> bool: ...
    def GetNumCtx(self) -> int: ...
    def EnsureContextBudget(self, messages, **kwargs) -> Any: ...
    def Chat(self, messages, **kwargs) -> Any: ...
    def ChatStream(self, messages, **kwargs) -> Any: ...
    def close(self) -> None: ...


def create_provider(config: RuntimeConfig, tools: ToolHandle) -> ModelProvider:
    # Lazy import: fake providers do not require the OpenAI SDK.
    from handlers.OllamaHandle import OllamaHandler
    return OllamaHandler(base_url=config.endpoint, key=config.api_key,
                         tool_handle=tools, request_timeout=config.request_timeout)
