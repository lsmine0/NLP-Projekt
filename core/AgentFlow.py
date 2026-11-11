from __future__ import annotations

import handlers.ContextHandle as Contx
from handlers.OllamaHandle import OllamaHandler
from core import Logger
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

DEFAULT_TOOL_CHARS = 1024
DEFAULT_TOOL_TIMEOUT = 10
DEFAULT_RESERVE_TOKENS = 8000


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
    ):
        # AgentFlow uses OllamaHandler directly instead of receiving it via constructor
        self.ollama = OllamaHandler(
            base_url=base_url,
            key=key,
            reserve_tokens=reserve_tokens,
            auto_budget=auto_budget,
        )
        self.tools = self.ollama.ToolsHandle

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
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self.tools.execute_tool, call["name"], call["args"])
                try:
                    raw_result = fut.result(timeout=DEFAULT_TOOL_TIMEOUT)
                except FutTimeout:
                    raw_result = json.dumps({
                        "success": False,
                        "error": f"Tool '{call['name']}' timed out after {DEFAULT_TOOL_TIMEOUT}s",
                        "tool": call["name"],
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

    def Run(self, user_input: str) -> str:
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
            messages_sent = self.context

            response = self.ollama.Chat(
                messages=messages_sent,
                max_tokens=self.reserve_tokens,
            )
            self._CalibrateTokens(messages_sent, response)

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
                        "args": json.loads(fn.arguments or "{}"),
                    })

            Contx.ContextHandle.Append(
                self.context,
                message.model_dump(exclude_none=True)
            )

            limit_reached = self._ExecuteToolCalls(calls)
            if limit_reached:
                return self._last_answer or ""

        return self._last_answer or ""

    def RunStream(self, user_input: str, as_events: bool = False):
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
            tool_calls = []
            last_chunk = None
            for chunk in self.ollama.ChatStream(
                messages=self.context,
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
                tc = getattr(delta, "tool_calls", None)
                if tc:
                    tool_calls.extend(tc)
            self._CalibrateTokens(self.context, last_chunk)
            answer = "".join(answer_parts)
            if tool_calls:
                tool_call_dicts = []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        tool_call_dicts.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": fn.name or "",
                                "arguments": fn.arguments or "{}",
                            },
                        })
                Contx.ContextHandle.Append(
                    self.context,
                    {"role": "assistant", "content": answer, "tool_calls": tool_call_dicts},
                )
                calls = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"] or "{}"),
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
