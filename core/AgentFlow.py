from __future__ import annotations

import handlers.ContextHandle as Contx
from core.ModelProvider import ModelProvider, create_provider
from core.RuntimeConfig import RuntimeConfig
from handlers.ToolHandle import ToolHandle, use_tool_session
from core import Logger
import json
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

DEFAULT_TOOL_CHARS = 1024
DEFAULT_TOOL_TIMEOUT = 10
DEFAULT_RESERVE_TOKENS = 8000


def _ParseToolArguments(raw) -> dict:
    """Zgoda na argumenty toola: poprawny JSON, dict albo ucięty fragment (#51).

    Strumienie OpenAI-compatible (np. llama.cpp) wysyłają `arguments` kawałkami.
    Po scaleniu powinien być poprawnym JSONem; gdy model urwał generowanie w
    połowie obiektu, zamykanie całego turnu błędem jest gorsze niż pusty zestaw
    argumentów - tool z opcjonalnymi parametrami (inspect_image) dalej działa.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        Logger.printWarning(
            f"Argumenty toola nie są poprawnym JSONem ({text[:40]!r}) - traktuję jako {{}}."
        )
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _MergeToolCallFragment(slots: dict, order: list, fragment) -> None:
    """Scal jeden fragment strumienia `tool_calls` w cały call (#51).

    Kawałki tego samego wywołania przychodzą z tym samym `index`, a id/nazwa są
    tylko w pierwszym z nich. Traktowanie każdego fragmentu jako osobnego toola
    (stare `tool_calls.extend(tc)`) dawało ucięty JSON i `json.JSONDecodeError`
    - cały turn ginął jako "Błąd modelu.".
    """
    function = getattr(fragment, "function", None)
    name_piece = (getattr(function, "name", "") or "") if function is not None else ""
    args_piece = (getattr(function, "arguments", "") or "") if function is not None else ""
    if isinstance(args_piece, dict):  # server zwraca obiekt zamiast stringu
        args_piece = json.dumps(args_piece)
    call_id = getattr(fragment, "id", None) or ""

    index = getattr(fragment, "index", None)
    if isinstance(index, bool) or not isinstance(index, int):
        # Server bez `index`: nowy call rozpoznajemy po id albo nazwie.
        index = len(order) if (call_id or name_piece) else (order[-1] if order else 0)

    slot = slots.get(index)
    if slot is None:
        slots[index] = {
            "id": call_id or f"call_{index}",
            "type": "function",
            "function": {"name": name_piece, "arguments": args_piece},
        }
        order.append(index)
        return

    if call_id and not slot["id"]:
        slot["id"] = call_id
    if name_piece:
        slot["function"]["name"] += name_piece  # nazwa też może być strugana
    if args_piece:
        slot["function"]["arguments"] += args_piece


class AgentFlow:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        key: str = "ollam",
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
        reserve_tokens: int = 1024,
        max_tool_calls: int = 15,
        auto_budget: bool = True,
        *,
        tool_handle: ToolHandle | None = None,
        provider: ModelProvider | None = None,
        message_resolver: Optional[Callable[[list], list]] = None,
    ):
        if provider is not None and tool_handle is None:
            tool_handle = provider.ToolsHandle
        self.tools = tool_handle if tool_handle is not None else ToolHandle()
        if provider is not None and provider.ToolsHandle is not self.tools:
            raise ValueError('Provider i agent muszą używać tego samego rejestru narzędzi.')
        self.ollama = provider if provider is not None else create_provider(
            RuntimeConfig(endpoint=base_url, api_key=key), self.tools
        )


        if provider is None:
            self.ollama.reserve_tokens = max(256, int(reserve_tokens))
            self.ollama.auto_budget = bool(auto_budget)

        self.num_ctx = 0
        self.reserve_tokens = max(256, int(reserve_tokens))
        self.max_context_tokens = 2048
        self.max_tool_calls = max(1, int(max_tool_calls))

        self.context: list = []
        if system_prompt:
            Contx.ContextHandle.Append(
                self.context, Contx.ContextHandle.Create(system_prompt, role="system")
            )

        self.max_iterations = max(1, int(max_iterations))

        self._token_ratio = 1.0
        self._last_answer: Optional[str] = None
        self._total_tool_calls_count = 0

        # #51 (obrazy/załączniki):
        # - message_resolver: zamienia referencje {'type':'attachment','id':...} na parts
        #   image_url z data URL. Wywoływany TYLKO przy budowaniu payloadu, więc w
        #   self.context nigdy nie lądują base64 ani ścieżki z dysku.
        # - session_scope: {'chat_id', 'attachment_store'} przekazywane narzędziom
        #   (tools/ImageTools.py) przez handlers.ToolHandle.current_tool_session().
        self.message_resolver = message_resolver
        self.session_scope: Optional[dict] = None

        Logger.printInfo(
            f"AgentFlow init: num_ctx={self.num_ctx}, reserve={self.reserve_tokens}, "
            f"max_context={self.max_context_tokens} tokens, iter <= {self.max_iterations}"
        )
    def RefreshModelContext(self) -> None:
        """Refresh context limits from the currently loaded Ollama model."""

        num_ctx = int(self.ollama.GetNumCtx())

        if num_ctx <= 0:
            raise RuntimeError("Invalid context size reported by Ollama.")

        changed = num_ctx != self.num_ctx

        self.num_ctx = num_ctx
        self.reserve_tokens = max(int(num_ctx * 0.5), DEFAULT_RESERVE_TOKENS)
        self.max_context_tokens = max(2048, self.num_ctx - self.reserve_tokens)

        if changed:
            Logger.printInfo(
                f"Model context initialized: num_ctx={self.num_ctx}, "
                f"reserve={self.reserve_tokens}, "
                f"max_context={self.max_context_tokens}"
            )

    def _SummarizeOldHistory(self, old_messages: list) -> Optional[str]:
        summary_budget = max(1024, int(self.num_ctx * 7 // 8))
        prefix, _ = Contx.ContextHandle.TrimHard(old_messages, summary_budget)
        try:
            response = self.ollama.Chat(
                messages=[
                    Contx.ContextHandle.Create(
                        "Stary fragment rozmowy przed ostatnim pytaniem użytkownika. "
                        "Zachowaj najważniejsze fakty, nazwy narzędzi i ich wyniki, otwarte wątki. "
                        "Odpowiedz wyłącznie podsumowaniem, bez nowych informacji.",
                        role="system",
                    ),
                ] + prefix,
                max_tokens=max(128, min(self.reserve_tokens, self.num_ctx // 8)),
                temperature=0.2,
                auto_budget=False,
            )
        except Exception as exc:
            Logger.printError(f"Błąd podsumowania: {exc}")
            return None
        return (response.choices[0].message.content or "").strip() or None

    def _CalibrateTokens(self, messages: list, response) -> None:
        try:
            usage = getattr(response, "usage", None)
            if not usage:
                return
            actual_prompt = getattr(usage, "prompt_tokens", None)
            actual_completion = getattr(usage, "completion_tokens", None)
        except Exception:
            return

        if not actual_prompt:
            return

        est = Contx.ContextHandle.HistoryTokens(messages)
        if not est:
            return

        raw_ratio = actual_prompt / est
        ratio = max(0.5, min(3.0, raw_ratio))
        self._token_ratio = (self._token_ratio + ratio) / 2
        
        comp_info = f", completion_tokens={actual_completion}" if actual_completion is not None else ""
        Logger.printInfo(
            f"Korekta estymacji tokenów: ratio={self._token_ratio:.2f} "
            f"(prompt_tokens={actual_prompt}{comp_info}, estymacja={est})"
        )

    def _ApiMessages(self, messages: list) -> list:
        """Kopia `messages` gotowa dla API (#51); oryginał NIE jest mutowany.

        Zwraca oryginalną listę, gdy resolver nie jest ustawiony (message text-only).
        Błąd resolvera (np. AttachmentError – załącznik z innego czatu) propaguje do
        AgentAdapter, który zamienia go na czytelny event 'error'.
        """
        resolver = self.message_resolver
        if resolver is None:
            return messages
        resolved = resolver(messages)
        return resolved if isinstance(resolved, list) else messages

    def _RunToolWithScope(self, name: str, args: dict) -> str:
        """Woła tool z zakresem sesji ustawionym w tym samym wątku (#51).

        Wątki ThreadPoolExecutor nie widzą ContextVarów ustawionych w wątku
        requestowym, dlatego scope musi być ustawiony inside this callable.
        """
        with use_tool_session(self.session_scope):
            return self.tools.execute_tool(name, args)

    def _ExecuteToolCalls(self, calls: list) -> bool:
        for call in calls:
            self._total_tool_calls_count += 1
            if self._total_tool_calls_count > self.max_tool_calls:
                Logger.printWarning(
                    f"Przekroczono maksymalny limit wywołań narzędzi ({self.max_tool_calls}). Zatrzymuję agenta."
                )
                return True

            Logger.printInfo(
                f"Executing tool: {call['name']} (Wywołanie {self._total_tool_calls_count}/{self.max_tool_calls})"
            )

            # ponytail: per-call thread pool for the timeout; fine for a handful of tool calls per run
            # #51: _RunToolWithScope (nie execute_tool) - setuje session scope w wątku workera.
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._RunToolWithScope, call["name"], call["args"])
                try:
                    raw_result = fut.result(timeout=DEFAULT_TOOL_TIMEOUT)
                except FutTimeout:
                    raw_result = json.dumps({
                        "success": False,
                        "error": f"Tool '{call['name']}' timed out after {DEFAULT_TOOL_TIMEOUT}s",
                        "result": None,
                    }, ensure_ascii=False)

            result, truncated = Contx.ContextHandle.TruncateJson(
                raw_result, DEFAULT_TOOL_CHARS // 4
            )
            if truncated:
                Logger.printWarning(
                    f"Wynik toola '{call['name']}' ograniczony do {DEFAULT_TOOL_CHARS} "
                    f"znaków (oznaczenie 'truncated')."
                )

            Contx.ContextHandle.Add(
                self.context, result, role="tool", tool_call_id=call["id"]
            )
        return False

    def Run(self, user_input) -> str:
        """Non-streaming run. `user_input` may be str or list multimodal parts (#51)."""
        self.RefreshModelContext()
        
        self._total_tool_calls_count = 0
        Contx.ContextHandle.Add(
            self.context,
            user_input,
            role="user"
        )

        for iteration in range(self.max_iterations):
            Logger.printInfo(f"Agent iteration: {iteration + 1}")

            self.ollama.EnsureContextBudget(
                self.context,
                summarizer=self._SummarizeOldHistory,
                ratio=self._token_ratio,
                reserve_tokens=self.reserve_tokens,
            )
            # Budżet liczymy na referencjach (taniej), w payload trafia kopia z data URL.
            api_messages = self._ApiMessages(self.context)

            response = self.ollama.Chat(
                messages=api_messages,
                max_tokens=self.reserve_tokens,
            )
            # Kalibracja ratio po referencjach, nie po megabajtach base64 (#51).
            self._CalibrateTokens(self.context, response)

            message = response.choices[0].message

            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                Logger.printThinking(reasoning)

            if not message.tool_calls:
                answer = message.content or ""
                self._last_answer = answer
                Contx.ContextHandle.Add(
                    self.context,
                    answer,
                    role="assistant"
                )
                return answer

            calls = []
            for tc in message.tool_calls:
                fn = getattr(tc, "function", None)
                if fn is not None:
                    calls.append({
                        "id": tc.id,
                        "name": fn.name,
                        "args": _ParseToolArguments(fn.arguments),
                    })

            Contx.ContextHandle.Append(
                self.context,
                message.model_dump(exclude_none=True)
            )

            limit_reached = self._ExecuteToolCalls(calls)
            if limit_reached:
                return self._last_answer or ""

        return self._last_answer or ""

    def RunStream(self, user_input, as_events: bool = False):
        """Streaming run. `user_input` may be str or list multimodal parts (#51)."""
        self.RefreshModelContext()
        self._total_tool_calls_count = 0
        Contx.ContextHandle.Add(self.context, user_input, role="user")
        for iteration in range(self.max_iterations):
            Logger.printInfo(f"Agent iteration (stream): {iteration + 1}")
            self.ollama.EnsureContextBudget(
                self.context,
                summarizer=self._SummarizeOldHistory,
                ratio=self._token_ratio,
                reserve_tokens=self.reserve_tokens,
            )
            answer_parts = []
            # #51: kawałki tool_calls scalane po `index` (slots/order), nie w płaską
            # listę - inaczej arguments przychodzą ucięte i parse crashuje turn.
            tool_slots: dict[int, dict] = {}
            tool_order: list[int] = []
            last_chunk = None
            # #51: payload = kopia z data URL; self.context zostaje referencjami (ID).
            for chunk in self.ollama.ChatStream(
                messages=self._ApiMessages(self.context),
                max_tokens=self.reserve_tokens,
                auto_budget=False,
            ):
                last_chunk = chunk
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {"type": "thinking", "text": reasoning} if as_events else reasoning
                text = delta.content
                if text:
                    yield {"type": "token", "text": text} if as_events else text
                    answer_parts.append(text)
                fragments = getattr(delta, "tool_calls", None)
                if fragments:
                    for fragment in fragments:
                        _MergeToolCallFragment(tool_slots, tool_order, fragment)
            self._CalibrateTokens(self.context, last_chunk)
            answer = "".join(answer_parts)
            tool_call_dicts = [tool_slots[key] for key in tool_order]
            if tool_call_dicts:
                Contx.ContextHandle.Append(
                    self.context,
                    {"role": "assistant", "content": answer, "tool_calls": tool_call_dicts},
                )
                calls = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": _ParseToolArguments(tc["function"]["arguments"]),
                    }
                    for tc in tool_call_dicts
                ]
                limit_reached = self._ExecuteToolCalls(calls)
                if limit_reached:
                    return
            else:
                self._last_answer = answer
                Contx.ContextHandle.Add(self.context, answer, role="assistant")
                return
