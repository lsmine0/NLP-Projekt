from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional
from openai import OpenAI
from handlers.ToolHandle import ToolHandle, short_tool_description
import handlers.ContextHandle as Contx
import requests
from core import Logger

DEFAULT_NUM_CTX = 16384


class OllamaHandler:
    def __init__(
        self,
        base_url="http://127.0.0.1:11434/v1",
        key="ollam",
        reserve_tokens: int = 1024,
        auto_budget: bool = True,
        *,
        tool_handle: ToolHandle | None = None,
        request_timeout: float = 60.0,
    ):
        if "://" not in base_url:
            base_url = f"http://{base_url}"
        # localhost -> 127.0.0.1: na Windowsie "localhost" najpierw próbuje IPv6 (::1),
        # na którym Ollama nie słucha, co dokłada ~2 s do każdego requestu.
        base_url = re.sub(r"^(\w+://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", base_url)
        self.base_url = base_url.rstrip("/").removesuffix("/v1") + "/v1"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=key, timeout=request_timeout, max_retries=0
        )
        self.request_timeout = request_timeout
        self.model = None
        self.thinking = None
        # #51: jedna probe /api/show per model (capabilities + num_ctx), kasowana w SetModel.
        self._model_info_cache: dict[str, Any] = {}
        self.ToolsHandle = tool_handle
        self.reserve_tokens = max(256, int(reserve_tokens))
        self.auto_budget = bool(auto_budget)

    def close(self) -> None:
        self.client.close()

    def GetModels(self):
        Logger.printInfo("Getting model list.")
        return self.client.models.list()

    def SetModel(self, ModelName):
        if ModelName in [model.id for model in self.GetModels()]:
            # #51: stary cache metadanych (capabilities/vision, num_ctx) jest nieaktualny.
            self._model_info_cache.clear()
            self.model = ModelName
            Logger.printInfo(f"Model {ModelName} set on endpoint.")
            return True
        else: 
            Logger.printWarning(f"model {ModelName} not available on this Ollama endpoint")
            return False
        
    def GetModelInfo(self):
        """Metadane modelu z cache'em (#51): odpytujemy /api/show raz per model.

        Cache'ujemy również wynik pusty – powtórne odpytywanie przy każdej iteracji
        flow dokładałoby round-trip (patrz reports/DIAGNOZA_CZASU.md). SetModel()
        kasuje cache, więc probe powtarza się po zmianie modelu.
        """
        if not self.model:
            Logger.printError("Get info on Invalid model.")
            return {}
        cached = self._model_info_cache.get(self.model)
        if cached is not None:
            return cached
        info = self._FetchModelInfo()
        if not isinstance(info, dict):
            info = {}
        self._model_info_cache[self.model] = info
        return info

    def _FetchModelInfo(self):
            
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
                timeout=min(3.0, self.request_timeout),
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            pass
        try:
            response = requests.get(
                f"{native_url}/props",
                headers=headers,
                timeout=min(3.0, self.request_timeout),
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

    def GetCapabilities(self) -> list:
        """Lista capabilities z /api/show (z cache'em); [] gdy server jej nie podał."""
        info = self.GetModelInfo()
        capabilities = info.get("capabilities") if isinstance(info, dict) else None
        return list(capabilities) if isinstance(capabilities, (list, tuple)) else []      

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

    def supports_vision(self) -> Optional[bool]:
        """Czy model przyjmuje obrazki (vision): True/False albo None (nieustalable).

        Tri-state (#51): jawne False = wysyłka obrazu blokowana z czytelny błędem;
        None = provider nie podał capabilities (/props fallback, server bez
        metadanych) – adapter traktuje to jako allow, żeby nie penalizować
        false-negativów.
        """
        if not self.model:
            return None
        info = self.GetModelInfo()
        if not isinstance(info, dict):
            return None
        capabilities = info.get("capabilities")
        if isinstance(capabilities, (list, tuple)) and capabilities:
            for capability in capabilities:
                text = str(capability).lower()
                if "vision" in text or "image" in text:
                    return True
            return False
        # Brak capabilities: szukamy hintów w model_info (projector/encoder wizualny).
        model_info = info.get("model_info")
        if isinstance(model_info, dict):
            for key in model_info:
                lowered = str(key).lower()
                if "vision" in lowered or "image_encoder" in lowered or "mmproj" in lowered:
                    return True
        return None

    def GetVision(self) -> bool:
        """Stary kontrakt bool: True tylko gdy vision byl potwierdzony."""
        return self.supports_vision() is True          
  
    def GetNumCtx(self) -> int:
        if not self.model:
            Logger.printWarning("No model selected - default num_ctx used.")
            return DEFAULT_NUM_CTX
        # Ollama: /api/ps podaje kontekst, z jakim model jest FAKTYCZNIE załadowany
        # (np. OLLAMA_CONTEXT_LENGTH serwera). Działa tylko, gdy model jest w pamięci.
        loaded_ctx = self._GetLoadedNumCtx()
        if loaded_ctx:
            return loaded_ctx
        info = self.GetModelInfo()
        if not isinstance(info, dict):
            return DEFAULT_NUM_CTX
        params = info.get("parameters")
        if isinstance(params, dict):
            n_ctx = params.get("num_ctx")
            if isinstance(n_ctx, (int, float)):
                return int(n_ctx)
        elif isinstance(params, str):
            for line in params.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].lower() == "num_ctx":
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
        for key, value in (info.get("options") or {}).items():
            if "num_ctx" in key.lower():
                try:
                    return int(str(value))
                except (TypeError, ValueError):
                    continue
        for key, value in (info.get("model_info") or {}).items():
            if key.endswith(".context_length"):
                try:
                    # To MAKSIMUM modelu (np. 262144 dla qwen3:4b), a nie kontekst,
                    # z którym serwer go uruchamia - nie zakładamy więcej niż domyślne.
                    return min(int(value), DEFAULT_NUM_CTX)
                except (TypeError, ValueError):
                    continue
        return DEFAULT_NUM_CTX

    def _GetLoadedNumCtx(self) -> Optional[int]:
        try:
            response = requests.get(f"{self.base_url.removesuffix('/v1')}/api/ps", timeout=2)
            response.raise_for_status()
            for loaded in response.json().get("models", []):
                if loaded.get("name") == self.model or loaded.get("model") == self.model:
                    ctx = loaded.get("context_length")
                    return int(ctx) if ctx else None
        except Exception:
            pass
        return None

    def Warmup(self) -> bool:
        """Ładuje model do pamięci (tylko Ollama), żeby pierwsze zapytanie nie płaciło za zimny start."""
        if not self.model:
            return False
        try:
            requests.post(
                f"{self.base_url.removesuffix('/v1')}/api/generate",
                json={"model": self.model},
                timeout=300,
            ).raise_for_status()
            Logger.printInfo(f"Model {self.model} warmed up.")
            return True
        except Exception as exc:
            Logger.printWarning(f"Warmup pominięty (serwer bez /api/generate?): {exc}")
            return False

    @staticmethod
    def _ShortDescription(value: Any, limit: int = 60) -> str:
        # Skracanie opisów narzędzi ma jedno źródło (handlers.ToolHandle):
        # schema wysyłana modelowi, widok UI i meta-narzędzie list_tools skracają
        # opisy identycznie, więc „krótki opis” znaczy wszędzie to samo.
        return short_tool_description(value, limit)

    @classmethod
    def _CompactSchema(cls, schema: Any, depth: int = 0) -> dict[str, Any]:
        """Keep structure/types but drop long descriptions from tool schemas."""
        if depth > 20 or not isinstance(schema, dict):
            return {}
        result: dict[str, Any] = {}
        for key in (
            "type", "enum", "default", "minimum", "maximum", "format",
            "pattern", "minItems", "maxItems", "uniqueItems",
        ):
            value = schema.get(key)
            if isinstance(value, (str, int, float, bool, list)) or value is None:
                if value is not None:
                    result[key] = value
        properties = schema.get("properties")
        if isinstance(properties, dict):
            result["properties"] = {
                str(name): cls._CompactSchema(item, depth + 1)
                for name, item in properties.items()
                if isinstance(item, dict)
            }
        required = schema.get("required")
        if isinstance(required, list):
            result["required"] = [str(item) for item in required if isinstance(item, str)]
        items = schema.get("items")
        if isinstance(items, dict):
            result["items"] = cls._CompactSchema(items, depth + 1)
        elif isinstance(items, list):
            result["items"] = [
                cls._CompactSchema(item, depth + 1)
                for item in items
                if isinstance(item, dict)
            ]
        for key in ("anyOf", "oneOf", "allOf"):
            value = schema.get(key)
            if isinstance(value, list):
                result[key] = [
                    cls._CompactSchema(item, depth + 1)
                    for item in value
                    if isinstance(item, dict)
                ]
        additional = schema.get("additionalProperties")
        if isinstance(additional, (bool, dict)):
            result["additionalProperties"] = (
                cls._CompactSchema(additional, depth + 1)
                if isinstance(additional, dict) else additional
            )
        return result

    @classmethod
    def _CompactTools(cls, tools: list) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}, "required": []}
            compact.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": cls._ShortDescription(function.get("description")),
                    "parameters": cls._CompactSchema(parameters),
                },
            })
        return compact

    def _EstimateToolTokens(self, tools: list | None = None) -> int:
        if tools is None:
            tools = self.ToolsHandle.get_tools_list() if self.ToolsHandle is not None else []
        if not tools:
            return 0
        try:
            serialized = json.dumps(self._CompactTools(tools), ensure_ascii=False)
        except (TypeError, ValueError):
            return 0
        return max(0, len(serialized) // 4)

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
        reserve = min(reserve, max(256, num_ctx // 2))
        tool_tokens = self._EstimateToolTokens()
        effective_ratio = max(0.5, min(3.0, float(ratio or 1.0)))
        available = num_ctx - reserve - 128 - tool_tokens
        budget = max(256, int(available / effective_ratio))
        history_tokens = Contx.ContextHandle.HistoryTokens(messages)
        Logger.printInfo(
            f"Context budget: num_ctx={num_ctx}, reserve={reserve}, "
            f"tool_tokens={tool_tokens}, message_tokens={history_tokens}, "
            f"budget={budget}"
        )
        if available <= 0:
            Logger.printWarning(
                "Kontekst modelu jest mniejszy niż rezerwa i schematy narzędzi; "
                "prompt może zostać przycięty."
            )

        # Wewnątrz budżetu - nic do robienia.
        if history_tokens <= budget:
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
            # Endpoint /v1 Ollamy ignoruje "think" - steruje się nim przez reasoning_effort.
            if self.thinking is False:
                extra_body["reasoning_effort"] = "none"
            elif isinstance(self.thinking, str):
                extra_body["reasoning_effort"] = self.thinking

        tools = self.ToolsHandle.get_tools_list() if self.ToolsHandle is not None else []
        if tools:
            # Full schemas with long parameter descriptions do not fit a 4096-token
            # model together with the system prompt. Types/structure are enough.
            kwargs["tools"] = self._CompactTools(tools)

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

    def ChatNoTools(self, messages, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
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