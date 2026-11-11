from __future__ import annotations

from typing import Any, Callable, Optional
from openai import OpenAI
from handlers import ToolHandle as Thandle
import handlers.ContextHandle as Contx
import requests
from core import Logger

DEFAULT_NUM_CTX = 16384


class OllamaHandler:
    def __init__(
        self,
        base_url="http://localhost:11434/v1",
        key="ollam",
        reserve_tokens: int = 1024,
        auto_budget: bool = True,
    ):
        if "://" not in base_url:
            base_url = f"http://{base_url}"
        self.base_url = base_url.rstrip("/").removesuffix("/v1") + "/v1"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=key
        )
        self.model = None
        self.thinking = None
        self.ToolsHandle = Thandle.ToolHandle()
        self.reserve_tokens = max(256, int(reserve_tokens))
        self.auto_budget = bool(auto_budget)

    def GetModels(self):
        Logger.printInfo("Getting model list.")
        return self.client.models.list()

    def SetModel(self, ModelName):
        if ModelName in [model.id for model in self.GetModels()]:
            self.model = ModelName
            Logger.printInfo(f"Model {ModelName} set on endpoint.")
            return True
        else: 
            Logger.printWarning(f"model {ModelName} not available on this Ollama endpoint")
            return False
        
    def GetModelInfo(self):
        if not self.model:
            Logger.printError("Get info on Invalid model.")
            return {}
            
        native_url = self.base_url.removesuffix("/v1")
        headers = {}
        if hasattr(self.client, "api_key") and self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        # 1. Try hitting the native Ollama endpoint first (/api/show)
        try:
            response = requests.post(
                f"{native_url}/api/show",
                json={"model": self.model},
                headers=headers,
                timeout=3,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            pass
        try:
            response = requests.get(
                f"{native_url}/props",
                headers=headers,
                timeout=3,
            )
            response.raise_for_status()
            props = response.json()
            
            # Extract context window from llama.cpp /props layout (e.g., n_ctx)
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx", DEFAULT_NUM_CTX)
            
            Logger.printInfo(f"Successfully retrieved model properties via GET /props (n_ctx={n_ctx}).")
            return {
                "parameters": {
                    "num_ctx": n_ctx
                }
            }
        except Exception as exc:
            Logger.printWarning(f"GET /props fallback failed, using default context: {exc}")
            return {}

    def GetCapabilities(self):
        return self.GetModelInfo().get("capabilities", [])      

    def GetThinking(self):
        return self.GetModelInfo().get("thinking")

    def SetThinking(self, level):
        thinking_info = self.GetThinking()

        if thinking_info is None:
            if level is not None:
                Logger.printError(f"Model does not support thinking.")
                return
            self.thinking = None
            return

        supported = thinking_info["values"]

        if level not in supported and level is not None:
            Logger.printWarning(f"Unsupported thinking level.")

        self.thinking = level

    def GetVision(self):
        return "vision" in self.GetCapabilities()          
  
    def GetNumCtx(self) -> int:
        if not self.model:
            Logger.printWarning("No model selected - default num_ctx used.")
            return DEFAULT_NUM_CTX
        info = self.GetModelInfo()
        if not isinstance(info, dict):
            return DEFAULT_NUM_CTX
        params = info.get("parameters") or {}
        n_ctx = params.get("num_ctx")
        if isinstance(n_ctx, (int, float)):
            return int(n_ctx)
        for key, value in (info.get("options") or {}).items():
            if "num_ctx" in key.lower():
                try:
                    return int(str(value))
                except (TypeError, ValueError):
                    continue
        for key, value in (info.get("model_info") or {}).items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return DEFAULT_NUM_CTX

    def EnsureContextBudget(
        self,
        messages: list,
        summarizer: Optional[Callable[[list], Optional[str]]] = None,
        ratio: float = 1.0,
        reserve_tokens: Optional[int] = None,
    ) -> None:
        if not messages or self.model is None:
            return
        num_ctx = self.GetNumCtx()
        reserve = max(256, int(reserve_tokens or self.reserve_tokens))
        effective_ratio = max(0.5, min(3.0, float(ratio or 1.0)))
        budget = max(1024, int((num_ctx - reserve - 128) / effective_ratio))

        # Wewnątrz budżetu - nic do robienia.
        if Contx.ContextHandle.HistoryTokens(messages) <= budget:
            return

        # Przebij budgeta - odzysk: najpierw podsumowanie, potem Trim.
        if summarizer is not None:
            # Podsumowanie jako domyślny odzysk (droższy, ale tańszy długoterminowo).
            summarized, fits = Contx.ContextHandle.Summarize(messages, budget, summarizer)
            messages[:] = summarized
            if fits:
                Logger.printInfo(
                    "Stara historia zastąpiona podsumowaniem - kontekst mieści się w budżecie."
                )
                return
            Logger.printWarning(
                "Podsumowanie nie wystarczyło (LLM zwróciło None / nie mieści się w budżecie) - "
                f"przechodzę do przycięcia (budżet: {budget} tokenów, num_ctx: {num_ctx})."
            )

        # Fallback: Trim (rzuci stare, zachowuje ogon).
        new_history, removed, over_budget = Contx.ContextHandle.Trim(messages, budget)
        messages[:] = new_history
        if removed:
            Logger.printWarning(
                f"Kontekst przycięty: -{removed} wiadomości "
                f"(budżet: {budget} tokenów, num_ctx: {num_ctx})."
            )

        if over_budget:
            new_history, removed2 = Contx.ContextHandle.TrimHard(messages, budget)
            messages[:] = new_history
            if removed2:
                Logger.printWarning(
                    f"Ostateczne przycięcie kontekstu: -{removed2} wiadomości "
                    f"(budżet: {budget} tokenów, num_ctx: {num_ctx})."
                )

    def _Prepare(self, kwargs: dict[str, Any]):
        if self.model is None:
            Logger.printError(f"No model selected")
            raise RuntimeError("No model selected")

        extra_body = dict(kwargs.pop("extra_body", {}) or {})

        if self.thinking is not None:
            extra_body["think"] = self.thinking

        tools = self.ToolsHandle.get_tools_list()
        if tools:
            kwargs["tools"] = tools

        Logger.printInfo(f"Sending request to {self.model}")
        return self.model, (extra_body or None), kwargs

    def Chat(self, messages, auto_budget: Optional[bool] = None, **kwargs):
        model, extra_body, kwargs = self._Prepare(kwargs)
        if auto_budget is None:
            auto_budget = self.auto_budget
        if auto_budget:
            self.EnsureContextBudget(messages)
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            extra_body=extra_body,
            **kwargs,
        )

    def ChatStream(self, messages, auto_budget: Optional[bool] = None, **kwargs):
        model, extra_body, kwargs = self._Prepare(kwargs)
        if auto_budget is None:
            auto_budget = self.auto_budget
        if auto_budget:
            self.EnsureContextBudget(messages)
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            extra_body=extra_body,
            **kwargs,
        )