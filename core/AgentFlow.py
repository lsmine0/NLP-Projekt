from __future__ import annotations

import handlers.ContextHandle as Contx
from core.ModelProvider import ModelProvider, create_provider
from core.RuntimeConfig import RuntimeConfig
from handlers.ToolHandle import ToolHandle, use_tool_session
from core import Logger
from core.AuditLog import AuditLog
from core.Interactions import (
    POLL_CANCELLED,
    POLL_PENDING,
    POLL_TIMEOUT,
    InteractionBroker,
    public_approval_event,
    public_question_event,
)
from core.ToolPolicy import PreparedToolCall, safe_arguments
import json
import re
import time
import uuid
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

DEFAULT_TOOL_CHARS = 1024
DEFAULT_TOOL_TIMEOUT = 10
DEFAULT_RESERVE_TOKENS = 8000
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)
_TOOL_RETRY_NUDGE = (
    "Odpowiedz teraz użytkownikowi na podstawie powyższych wyników narzędzi. "
    "Nie kończ tury samym procesem myślenia ani pustą odpowiedzią. "
    "Jeśli narzędzie zwróciło błąd, wyjaśnij go i zaproponuj kolejny krok."
)
_TOOL_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_TOOL_JSON_CORRECTION_NUDGE = (
    "Poprzednia odpowiedź opisała wywołanie narzędzia jako JSON w treści. "
    "Nie rób tego. Wywołaj narzędzie wyłącznie przez mechanizm tool_calls "
    "(function calling). Jeśli nie możesz, odpowiedz bez bloków JSON."
)


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
        interaction_broker: InteractionBroker | None = None,
        audit_log: AuditLog | None = None,
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
        # #risk/approvals: broker współdzielony przez GUI, CLI i API; audit jest
        # opcjonalny, żeby bezpośrednie użycie biblioteki nie tworzyło plików.
        self.interaction_broker = interaction_broker
        self.audit_log = audit_log
        self.run_id: str = uuid.uuid4().hex
        self._run_id_preassigned = False
        self._tool_limit_reached = False

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
        # Reserve must leave room for the prompt and tool schemas. A fixed 8000
        # on a 4096-token model makes the context budget negative and trims the
        # user message/tool instructions.
        reserve = max(256, min(DEFAULT_RESERVE_TOKENS, int(num_ctx * 0.25)))
        reserve = min(reserve, max(256, num_ctx // 2))
        self.reserve_tokens = reserve
        self.max_context_tokens = max(512, self.num_ctx - self.reserve_tokens)
        if hasattr(self.ollama, "reserve_tokens"):
            self.ollama.reserve_tokens = self.reserve_tokens

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

    def _RunPreparedWithScope(self, prepared: PreparedToolCall) -> str:
        with use_tool_session(self.session_scope):
            return self.tools.execute_prepared(prepared)

    def _ChatId(self) -> Optional[str]:
        return (self.session_scope or {}).get("chat_id")

    def _PrepareCall(self, name: str, args: dict) -> Optional[PreparedToolCall]:
        prepare = getattr(self.tools, "prepare_tool_call", None)
        if not callable(prepare):
            return None
        return prepare(name, args)

    def _ToolFields(self, prepared: Optional[PreparedToolCall], name: str = "") -> dict:
        if prepared is None:
            return {"tool": name, "source": "builtin", "server": None, "risk": []}
        return {
            "tool": prepared.name,
            "source": prepared.spec.source,
            "server": prepared.spec.server,
            "risk": list(prepared.risk),
        }

    def _AddToolResult(self, call: dict, raw_result: str) -> None:
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

    @staticmethod
    def _ToolSucceeded(raw_result: str) -> bool:
        try:
            payload = json.loads(raw_result)
        except (TypeError, ValueError):
            return True
        return bool(payload.get("success")) if isinstance(payload, dict) else True

    @staticmethod
    def _VisibleText(text: str | None) -> str:
        """Text left after removing encapsulated reasoning blocks."""
        if not isinstance(text, str):
            return ""
        text = _THINK_BLOCK_RE.sub("", text)
        text = _THINK_OPEN_RE.sub("", text)
        return text.strip()

    def _ExtractToolJson(self, text: str | None) -> tuple[str, dict] | None:
        """Detect a textual tool-call JSON written into the answer content."""
        if not isinstance(text, str) or "{" not in text:
            return None

        candidates: list = []
        fenced = _TOOL_JSON_FENCE_RE.search(text)
        if fenced:
            candidates.append(fenced.group(1))
        stripped = text.strip()
        if stripped.startswith("{"):
            candidates.append(stripped)

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _end = decoder.raw_decode(text[index:])
            except ValueError:
                continue
            candidates.append(candidate)

        for candidate in candidates:
            if isinstance(candidate, str):
                try:
                    obj = json.loads(candidate)
                except ValueError:
                    continue
            else:
                obj = candidate
            if not isinstance(obj, dict):
                continue
            function = obj.get("function") if isinstance(obj.get("function"), dict) else {}
            name = obj.get("tool") or obj.get("name") or function.get("name")
            if "arguments" in obj:
                arguments = obj.get("arguments")
            elif "parameters" in obj:
                arguments = obj.get("parameters")
            else:
                arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = None
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                continue
            get_spec = getattr(self.tools, "get_tool_spec", None)
            if callable(get_spec) and get_spec(name) is None:
                continue
            return name, arguments
        return None

    def _RetryToolJsonMessage(self, answer: str):
        """One corrective completion with tools enabled (non-streaming Run)."""
        chat = getattr(self.ollama, "Chat", None)
        if not callable(chat):
            return None
        messages = list(self._ApiMessages(self.context))
        # Nie wstawiamy pełnej błędnej odpowiedzi: zajmowałaby cenne tokeny
        # i często prowokowała model do dalszego rozpisywania JSON-a.
        messages.append(Contx.ContextHandle.Create(_TOOL_JSON_CORRECTION_NUDGE, role="system"))
        try:
            response = chat(messages=messages, max_tokens=self.reserve_tokens)
            return response.choices[0].message
        except Exception:
            Logger.printWarning("Ponowienie po tekstowym tool callu nie powiodło się.")
            return None

    def _RetryAfterTools(self) -> str:
        """One non-tool completion after a tool result produced no visible answer."""
        chat_no_tools = getattr(self.ollama, "ChatNoTools", None)
        if not callable(chat_no_tools):
            Logger.printWarning(
                "Provider nie obsługuje ChatNoTools; nie można ponowić odpowiedzi po narzędziu."
            )
            return ""
        messages = list(self._ApiMessages(self.context))
        messages.append(Contx.ContextHandle.Create(_TOOL_RETRY_NUDGE, role="system"))
        base = {"messages": messages, "max_tokens": self.reserve_tokens}
        try:
            # Wyłączamy thinking, jeśli provider to obsługuje. Gdy endpoint odrzuci
            # extra_body, ponawiamy zwykłym wywołaniem bez narzędzi.
            try:
                response = chat_no_tools(
                    **base,
                    extra_body={"think": False, "reasoning_effort": "none"},
                )
            except Exception:
                response = chat_no_tools(**base)
            content = response.choices[0].message.content
        except Exception:
            Logger.printWarning("Ponowienie odpowiedzi po użyciu narzędzia nie powiodło się.")
            return ""
        if not isinstance(content, str):
            return ""
        return content if self._VisibleText(content) else ""

    def _RunWithTimeout(self, function, *args) -> tuple[bool, str | None]:
        """Run a tool without waiting for a hung worker after the timeout."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(function, *args)
            try:
                return True, future.result(timeout=DEFAULT_TOOL_TIMEOUT)
            except FutTimeout:
                return False, None
        finally:
            # Do not block the agent/UI on a stuck network call. The worker may
            # finish in the background; its result is intentionally ignored.
            executor.shutdown(wait=False, cancel_futures=True)

    def _AuditDecision(
        self,
        prepared: PreparedToolCall,
        *,
        decision: str,
        reason: str,
        approval_id: str | None = None,
        code: str | None = None,
    ) -> None:
        if self.audit_log is None:
            return
        self.audit_log.decision(
            tool=prepared.name,
            risk=prepared.risk,
            decision=decision,
            reason=reason,
            chat_id=self._ChatId(),
            run_id=self.run_id,
            approval_id=approval_id,
            source=prepared.spec.source,
            policy=prepared.policy,
            code=code,
        )

    def _AuditDestructive(
        self,
        prepared: PreparedToolCall,
        *,
        phase: str,
        approval_id: str | None = None,
        decision: str | None = None,
        success: bool | None = None,
        duration_ms: int | None = None,
    ) -> None:
        if self.audit_log is None or not prepared.spec.destructive:
            return
        self.audit_log.destructive(
            tool=prepared.name,
            risk=prepared.risk,
            phase=phase,
            chat_id=self._ChatId(),
            run_id=self.run_id,
            approval_id=approval_id,
            decision=decision,
            success=success,
            duration_ms=duration_ms,
        )

    def _WaitForInteraction(self, pending, event_type: str):
        """Yield heartbeats while waiting; return the terminal decision/answer."""
        while True:
            state = self.interaction_broker.poll(pending)
            if state == POLL_PENDING:
                yield {
                    "type": event_type,
                    "id": pending.id,
                    "expires_in": pending.expires_in,
                }
                time.sleep(0.5)
                continue
            return state

    def _DeniedTool(
        self,
        prepared: PreparedToolCall,
        code: str,
        message: str,
        *,
        approval_id: str | None = None,
        reason: str = "policy",
    ) -> str:
        self._AuditDecision(
            prepared,
            decision="deny",
            reason=reason,
            approval_id=approval_id,
            code=code,
        )
        return self.tools.denial_json(
            prepared, code, message, approval_id=approval_id
        )

    def _HandleQuestion(self, prepared: PreparedToolCall, interactive: bool):
        """Yield interaction events and return a JSON tool result."""

        arguments = prepared.arguments
        question = arguments.get("question")
        raw_options = arguments.get("options")
        header = arguments.get("header") or ""
        allow_custom = arguments.get("allow_custom", True)

        if not isinstance(question, str) or not question.strip():
            raw = json.dumps(
                {"success": False, "error": "Pytanie nie może być puste.", "result": None},
                ensure_ascii=False,
            )
            yield {
                "type": "tool_result",
                "status": "error",
                **self._ToolFields(prepared),
            }
            return raw

        options = []
        if isinstance(raw_options, (list, tuple)):
            for item in raw_options:
                if isinstance(item, str) and item.strip():
                    options.append(item.strip())
        if len(options) < 2:
            raw = json.dumps(
                {
                    "success": False,
                    "error": "Narzędzie ask_user wymaga od 2 do 6 niepustych opcji.",
                    "result": None,
                },
                ensure_ascii=False,
            )
            yield {
                "type": "tool_result",
                "status": "error",
                **self._ToolFields(prepared),
            }
            return raw
        options = options[:6]

        if not interactive or self.interaction_broker is None:
            raw = self.tools.denial_json(
                prepared,
                "question_unavailable",
                "Pytanie do użytkownika nie jest dostępne w trybie niestrumieniowym.",
            )
            yield {
                "type": "tool_result",
                "status": "denied",
                "code": "question_unavailable",
                **self._ToolFields(prepared),
            }
            return raw

        safe = safe_arguments(prepared.spec, prepared.arguments)
        safe_question = str(safe.get("question") or question).strip()
        safe_options = [
            str(item) for item in (safe.get("options") or options) if str(item).strip()
        ] or options
        safe_header = str(safe.get("header") or header).strip()

        yield {"type": "tool_start", **self._ToolFields(prepared)}
        pending = self.interaction_broker.register_question(
            question=safe_question,
            options=safe_options,
            header=safe_header,
            allow_custom=bool(allow_custom),
            chat_id=self._ChatId(),
            run_id=self.run_id,
        )
        yield public_question_event(pending)
        state = yield from self._WaitForInteraction(pending, "question_pending")

        if isinstance(state, dict) and not state.get("cancelled"):
            answer = str(state.get("answer") or "").strip()
            option = state.get("option")
            result: dict = {"answer": answer}
            if option is not None:
                result["option"] = option
            raw = json.dumps(
                {"success": True, "error": None, "result": result},
                ensure_ascii=False,
            )
            yield {
                "type": "tool_result",
                "status": "ok",
                **self._ToolFields(prepared),
            }
            return raw

        code = "question_timeout" if state == POLL_TIMEOUT else "question_cancelled"
        raw = json.dumps(
            {
                "success": False,
                "error": "Użytkownik nie odpowiedział.",
                "result": {"cancelled": True, "code": code},
            },
            ensure_ascii=False,
        )
        yield {
            "type": "tool_result",
            "status": "denied",
            "code": code,
            **self._ToolFields(prepared),
        }
        return raw

    def _ExecuteToolCalls(self, calls: list, *, interactive: bool = False):
        """Execute calls, yielding approval/question/tool events as they happen.

        ``interactive=False`` (non-streaming runs) denies ``ask`` calls instead of
        waiting for a user that cannot answer.
        """

        self._tool_limit_reached = False
        for call in calls:
            self._total_tool_calls_count += 1
            if self._total_tool_calls_count > self.max_tool_calls:
                Logger.printWarning(
                    f"Przekroczono maksymalny limit wywołań narzędzi ({self.max_tool_calls}). Zatrzymuję agenta."
                )
                self._tool_limit_reached = True
                return

            Logger.printInfo(
                f"Executing tool: {call['name']} (Wywołanie {self._total_tool_calls_count}/{self.max_tool_calls})"
            )

            prepared = self._PrepareCall(call["name"], call["args"])
            if prepared is not None:
                denial_code = (prepared.denial or {}).get("code") if prepared.denial else None
                Logger.printInfo(
                    f"Tool prepare: {prepared.name} policy={prepared.policy} "
                    f"requires_approval={prepared.requires_approval} "
                    f"denial={denial_code or '-'}"
                )
            if prepared is None:
                # Backwards-compatible path for lightweight fake registries used
                # by tests/library callers that expose only execute_tool().
                yield {
                    "type": "tool_start",
                    "tool": call["name"],
                    "source": "builtin",
                    "server": None,
                    "risk": [],
                }
                completed, raw_result = self._RunWithTimeout(
                    self._RunToolWithScope, call["name"], call["args"]
                )
                if not completed or raw_result is None:
                    Logger.printWarning(
                        f"Tool '{call['name']}' timed out after {DEFAULT_TOOL_TIMEOUT}s"
                    )
                    raw_result = json.dumps({
                        "success": False,
                        "error": f"Tool '{call['name']}' timed out after {DEFAULT_TOOL_TIMEOUT}s",
                        "result": None,
                    }, ensure_ascii=False)
                yield {
                    "type": "tool_result",
                    "tool": call["name"],
                    "source": "builtin",
                    "server": None,
                    "risk": [],
                    "status": "ok" if self._ToolSucceeded(raw_result) else "error",
                }
                self._AddToolResult(call, raw_result)
                continue

            if prepared.denial is not None:
                code = str(prepared.denial.get("code", "denied"))
                message = str(prepared.denial.get("message", "Narzędzie nie zostało wykonane."))
                reason = (
                    "validation"
                    if code in ("unknown_tool", "invalid_arguments", "path_outside_allowed", "protected_path")
                    else "policy"
                )
                raw_result = self._DeniedTool(
                    prepared, code, message, reason=reason
                )
                yield {
                    "type": "tool_result",
                    "status": "denied",
                    "code": code,
                    **self._ToolFields(prepared),
                }
                self._AddToolResult(call, raw_result)
                continue

            if prepared.spec.kind == "user_question":
                raw_result = yield from self._HandleQuestion(prepared, interactive)
                self._AddToolResult(call, raw_result)
                continue

            approval_id: str | None = None
            decision = "allow"
            decision_reason = "policy"

            if prepared.requires_approval:
                if not interactive or self.interaction_broker is None:
                    raw_result = self._DeniedTool(
                        prepared,
                        "approval_unavailable",
                        "Zgoda użytkownika nie jest dostępna w trybie niestrumieniowym.",
                        reason="approval_unavailable",
                    )
                    yield {
                        "type": "tool_result",
                        "status": "denied",
                        "code": "approval_unavailable",
                        **self._ToolFields(prepared),
                    }
                    self._AddToolResult(call, raw_result)
                    continue

                chat_id = self._ChatId()
                if self.interaction_broker.session_allowed(chat_id, prepared.name):
                    decision = "allow_session"
                    decision_reason = "session"
                else:
                    pending = self.interaction_broker.register_approval(
                        tool=prepared.name,
                        risk=prepared.risk,
                        arguments=safe_arguments(prepared.spec, prepared.arguments),
                        chat_id=chat_id,
                        run_id=self.run_id,
                        source=prepared.spec.source,
                        server=prepared.spec.server,
                    )
                    approval_id = pending.id
                    Logger.printInfo(
                        f"Approval required: {prepared.name} risk={list(prepared.risk)}"
                    )
                    yield public_approval_event(pending)
                    terminal = yield from self._WaitForInteraction(
                        pending, "approval_pending"
                    )
                    if terminal not in ("allow_once", "allow_session"):
                        if terminal == POLL_TIMEOUT:
                            code, reason = "approval_timeout", "timeout"
                        elif terminal == POLL_CANCELLED:
                            code, reason = "client_disconnected", "client_disconnected"
                        else:
                            code, reason = "user_denied", "user"
                        raw_result = self._DeniedTool(
                            prepared,
                            code,
                            "Narzędzie nie zostało wykonane: użytkownik nie wyraził zgody.",
                            approval_id=approval_id,
                            reason=reason,
                        )
                        yield {
                            "type": "tool_result",
                            "status": "denied",
                            "code": code,
                            "approval_id": approval_id,
                            **self._ToolFields(prepared),
                        }
                        self._AddToolResult(call, raw_result)
                        continue
                    decision = terminal
                    decision_reason = "user" if terminal == "allow_once" else "user_session"
                    Logger.printInfo(
                        f"Approval decision: {prepared.name} -> {decision}"
                    )
                    self._AuditDecision(
                        prepared,
                        decision=decision,
                        reason=decision_reason,
                        approval_id=approval_id,
                    )

            yield {
                "type": "tool_start",
                "approval_id": approval_id,
                **self._ToolFields(prepared),
            }
            started = time.monotonic()
            self._AuditDestructive(
                prepared,
                phase="start",
                approval_id=approval_id,
                decision=decision,
            )
            completed, raw_result = self._RunWithTimeout(
                self._RunPreparedWithScope, prepared
            )
            if not completed or raw_result is None:
                Logger.printWarning(
                    f"Tool '{call['name']}' timed out after {DEFAULT_TOOL_TIMEOUT}s"
                )
                raw_result = json.dumps({
                    "success": False,
                    "error": f"Tool '{call['name']}' timed out after {DEFAULT_TOOL_TIMEOUT}s",
                    "result": None,
                }, ensure_ascii=False)
            duration_ms = int((time.monotonic() - started) * 1000)
            success = self._ToolSucceeded(raw_result)
            self._AuditDestructive(
                prepared,
                phase="end",
                approval_id=approval_id,
                decision=decision,
                success=success,
                duration_ms=duration_ms,
            )
            Logger.printInfo(
                f"Tool result: {prepared.name} status={'ok' if success else 'error'} "
                f"duration_ms={duration_ms}"
            )
            yield {
                "type": "tool_result",
                "status": "ok" if success else "error",
                "approval_id": approval_id,
                "duration_ms": duration_ms,
                **self._ToolFields(prepared),
            }
            self._AddToolResult(call, raw_result)

    def Run(self, user_input) -> str:
        """Non-streaming run. `user_input` may be str or list multimodal parts (#51)."""
        if not self._run_id_preassigned:
            self.run_id = uuid.uuid4().hex
        self._run_id_preassigned = False
        self.RefreshModelContext()
        
        self._total_tool_calls_count = 0
        saw_tool_result = False
        empty_retry_done = False
        tool_json_retry_done = False
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
                detected = self._ExtractToolJson(answer)
                if detected and not tool_json_retry_done:
                    tool_json_retry_done = True
                    Logger.printWarning(
                        f"Model opisał wywołanie narzędzia jako JSON (tool={detected[0]}); ponawiam."
                    )
                    corrected = self._RetryToolJsonMessage(answer)
                    if corrected is not None:
                        message = corrected
                elif detected:
                    Logger.printWarning(
                        f"Model nadal opisuje wywołanie narzędzia jako JSON (tool={detected[0]})."
                    )

            if not message.tool_calls:
                answer = message.content or ""
                if not self._VisibleText(answer) and saw_tool_result and not empty_retry_done:
                    empty_retry_done = True
                    Logger.printWarning(
                        "Model nie zwrócił odpowiedzi po użyciu narzędzia; ponawiam."
                    )
                    retry_answer = self._RetryAfterTools()
                    if retry_answer:
                        self._last_answer = retry_answer
                        Contx.ContextHandle.Add(self.context, retry_answer, role="assistant")
                        return retry_answer
                    Logger.printWarning("Ponowienie po narzędziu nie zwróciło odpowiedzi.")
                    self._last_answer = ""
                    return ""
                self._last_answer = answer
                if answer.strip():
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

            for _event in self._ExecuteToolCalls(calls, interactive=False):
                if _event.get("type") == "tool_result":
                    saw_tool_result = True
            if self._tool_limit_reached:
                return self._last_answer or ""

        return self._last_answer or ""

    def RunStream(self, user_input, as_events: bool = False):
        """Streaming run. `user_input` may be str or list multimodal parts (#51)."""
        if not self._run_id_preassigned:
            self.run_id = uuid.uuid4().hex
        self._run_id_preassigned = False
        self.RefreshModelContext()
        self._total_tool_calls_count = 0
        saw_tool_result = False
        empty_retry_done = False
        tool_json_retry_done = False
        tool_json_retry_pending = False
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
            api_messages = list(self._ApiMessages(self.context))
            if tool_json_retry_pending:
                api_messages.append(
                    Contx.ContextHandle.Create(_TOOL_JSON_CORRECTION_NUDGE, role="system")
                )
                tool_json_retry_pending = False
            for chunk in self.ollama.ChatStream(
                messages=api_messages,
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
            if not tool_call_dicts:
                detected = self._ExtractToolJson(answer)
                if detected and as_events:
                    if not tool_json_retry_done:
                        tool_json_retry_done = True
                        tool_json_retry_pending = True
                        Logger.printWarning(
                            f"Model opisał wywołanie narzędzia jako JSON "
                            f"(tool={detected[0]}); ponawiam z tool_calls."
                        )
                        yield {"type": "answer_reset", "reason": "tool_json"}
                        yield {
                            "type": "warning",
                            "message": (
                                "Model opisał wywołanie narzędzia w treści. "
                                "Ponawiam z użyciem tool_calls."
                            ),
                        }
                        continue
                    Logger.printWarning(
                        f"Model nadal opisuje wywołanie narzędzia jako JSON "
                        f"(tool={detected[0]})."
                    )
                    yield {"type": "answer_reset", "reason": "tool_json_failed"}
                    yield {
                        "type": "error",
                        "message": (
                            "Model nadal opisuje wywołanie narzędzia zamiast "
                            "używać tool_calls."
                        ),
                    }
                    return
                if detected:
                    Logger.printWarning(
                        f"Model opisał wywołanie narzędzia jako JSON "
                        f"(tool={detected[0]})."
                    )
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
                for event in self._ExecuteToolCalls(calls, interactive=as_events):
                    if event.get("type") == "tool_result":
                        saw_tool_result = True
                    if as_events:
                        yield event
                if self._tool_limit_reached:
                    return
            else:
                Logger.printInfo(
                    f"Model response: answer_chars={len(answer)}, "
                    f"tool_calls={len(tool_call_dicts)}"
                )
                if not self._VisibleText(answer) and saw_tool_result and not empty_retry_done:
                    empty_retry_done = True
                    Logger.printWarning(
                        "Model nie zwrócił odpowiedzi po użyciu narzędzia; ponawiam."
                    )
                    retry_answer = self._RetryAfterTools()
                    if retry_answer:
                        self._last_answer = retry_answer
                        Contx.ContextHandle.Add(self.context, retry_answer, role="assistant")
                        if as_events:
                            yield {"type": "token", "text": retry_answer}
                        else:
                            yield retry_answer
                        return
                    Logger.printWarning("Ponowienie po narzędziu nie zwróciło odpowiedzi.")
                    self._last_answer = ""
                    if as_events:
                        yield {
                            "type": "error",
                            "message": (
                                "Model nie zwrócił odpowiedzi po użyciu narzędzia. "
                                "Spróbuj ponownie."
                            ),
                        }
                    return
                self._last_answer = answer
                if answer.strip():
                    Contx.ContextHandle.Add(self.context, answer, role="assistant")
                return
