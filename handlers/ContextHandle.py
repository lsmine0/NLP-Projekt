from __future__ import annotations

import json
import math
from typing import Callable, List, Optional, Tuple


class ContextHandle:

    @staticmethod
    def Create(prompt, role="system", **kwargs) -> dict:
        return {"role": role, "content": prompt, **kwargs}

    @staticmethod
    def Add(history: list, prompt, pos=None, role="system", **kwargs) -> list:
        message = ContextHandle.Create(prompt, role, **kwargs)
        if pos is None:
            history.append(message)
        else:
            history.insert(pos, message)
        return history

    @staticmethod
    def Append(history: list, message: dict) -> list:
        history.append(message)
        return history

    @staticmethod
    def Update(history: list, prompt, pos=0) -> list:
        history[pos] = ContextHandle.Create(prompt)
        return history

    @staticmethod
    def Read(history, lista) -> str:
        """Wypisuje wybrane role jako tekst."""
        wybrane = ""
        for wiad in history:
            if wiad["role"] in lista:
                wybrane += f"{wiad['role']} : {wiad.get('content') or ''}\n"
        return wybrane

    # ================================================================
    # TOKENS - prosta heurystyka znakowa (znaki/4)
    # ================================================================
    @staticmethod
    def CountTokens(text: str) -> int:
        """Ilość tokenów w tekście (estymacja heurystyczna)."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def MessageTokens(message: dict) -> int:
        """Tokeny wiadomości + narzuty."""
        tokens = 5
        if message.get("content"):
            tokens += ContextHandle.CountTokens(message["content"])
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tokens += ContextHandle.CountTokens(fn.get("name") or "")
            tokens += ContextHandle.CountTokens(fn.get("arguments") or "")
        return tokens

    @staticmethod
    def HistoryTokens(history: list) -> int:
        """Tokeny całej historii."""
        return sum(ContextHandle.MessageTokens(m) for m in history)

    # --- helper wewnętrzne ---
    @staticmethod
    def _GroupMessages(history: list) -> List[list]:
        groups = []
        i, n = 0, len(history)
        while i < n:
            if history[i].get("role") == "assistant" and history[i].get("tool_calls"):
                group = [history[i]]
                j = i + 1
                while j < n and history[j].get("role") == "tool":
                    group.append(history[j])
                    j += 1
                groups.append(group)
                i = j
            else:
                groups.append([history[i]])
                i += 1
        return groups

    @staticmethod
    def _LastUserGroupIndex(groups: list) -> int:
        for idx in range(len(groups) - 1, -1, -1):
            if any(m.get("role") == "user" for m in groups[idx]):
                return idx
        return max(0, len(groups) - 1)

    # ================================================================
    # TRIM
    # ================================================================
    @staticmethod
    def Trim(history: list, max_tokens: int, reserve_tokens: int = 0) -> Tuple[list, int, bool]:
        if not history:
            return history, 0, False

        budget = int(max_tokens) - int(reserve_tokens or 0)
        if budget < 1:
            return history, 0, True
        groups = ContextHandle._GroupMessages(history)
        tail_idx = ContextHandle._LastUserGroupIndex(groups)

        tail_groups = groups[tail_idx:]
        system_tokens = sum(ContextHandle.MessageTokens(m) for m in groups[0])
        tail_tokens = sum(ContextHandle.MessageTokens(m) for g in tail_groups for m in g)

        if system_tokens + tail_tokens > budget:
            preserved = [m for m in groups[0]] + [m for g in tail_groups for m in g]
            return preserved, 0, True

        headroom = budget - system_tokens - tail_tokens
        kept_prefix = []
        for g in reversed(groups[1:tail_idx]):
            g_tokens = sum(ContextHandle.MessageTokens(m) for m in g)
            if g_tokens == 0 or g_tokens <= headroom:
                kept_prefix.append(g)
                headroom -= g_tokens
            else:
                break

        new_groups = groups[:1] + kept_prefix[::-1] + tail_groups
        new_history = [m for g in new_groups for m in g]
        removed = len(history) - len(new_history)
        over_budget = sum(ContextHandle.MessageTokens(m) for m in new_history) > budget
        return new_history, removed, over_budget

    @staticmethod
    def TrimHard(history: list, max_tokens: int) -> Tuple[list, int]:
        groups = ContextHandle._GroupMessages(history)
        if len(groups) <= 2:
            return list(history), 0
        budget = int(max_tokens)
        if budget < 1:
            return list(history), 0
        middle = groups[1:-1]
        removed = 0
        total = (
            sum(ContextHandle.MessageTokens(m) for m in groups[0])
            + sum(ContextHandle.MessageTokens(m) for m in groups[-1])
        )
        while total > budget and middle:
            g = middle.pop(0)
            total -= sum(ContextHandle.MessageTokens(m) for m in g)
            removed += len(g)
        new_history = [m for g in (groups[:1] + middle + groups[-1:]) for m in g]
        return new_history, removed

    # ================================================================
    # SUMMARIZE
    # ================================================================
    @staticmethod
    def Summarize(
        history: list,
        max_tokens: int,
        summarize_fn: Callable[[list], Optional[str]],
    ) -> Tuple[list, bool]:
        if not history:
            return history, False
        budget = int(max_tokens)
        groups = ContextHandle._GroupMessages(history)
        if len(groups) <= 2:
            return list(history), ContextHandle.HistoryTokens(history) <= budget

        def _summary_message(text: str) -> dict:
            return ContextHandle.Create(
                f"Stary fragment historii (zamiast skróconej wersji):\n{text}",
                role="system",
            )

        def _call_summarizer(prefix_msgs: list) -> Optional[str]:
            if not prefix_msgs:
                return None
            try:
                summary = summarize_fn(prefix_msgs)
            except Exception:
                return None
            if summary is None:
                return None
            summary = str(summary).strip()
            return summary or None

        # groups[0] is the original system prompt (initial prompt) - always preserved.
        sys_msgs = list(groups[0])

        tail_idx = ContextHandle._LastUserGroupIndex(groups)
        if tail_idx > 0:
            prefix_msgs = [m for g in groups[1:tail_idx] for m in g]
            summary = _call_summarizer(prefix_msgs)
            if summary is not None:
                tail_msgs = [m for g in groups[tail_idx:] for m in g]
                candidate = sys_msgs + [_summary_message(summary)] + tail_msgs
                if ContextHandle.HistoryTokens(candidate) <= budget:
                    return candidate, True

        if len(groups) > 2:
            mid_msgs = [m for g in groups[1:-1] for m in g]
            summary2 = _call_summarizer(mid_msgs)
            if summary2 is not None:
                last_msgs = list(groups[-1])
                candidate = sys_msgs + [_summary_message(summary2)] + last_msgs
                if ContextHandle.HistoryTokens(candidate) <= budget:
                    return candidate, True

        # LLM summary nie zadziałało (zwróciło None / błąd) albo nie mieści się w budżecie.
        # Fallback: Trim (rzuci stare, zachowuje ogon) - ta ścieżka jest domyślna dla odzysku budżetu.
        trimmed, removed, over_budget = ContextHandle.Trim(history, budget)
        return trimmed, not over_budget

    # ================================================================
    # TRUNCATE
    # ================================================================
    @staticmethod
    def TruncateJson(payload: str, max_tokens: int) -> Tuple[str, bool]:
        if not payload:
            return payload, False
        char_budget = max(64, int(max_tokens) * 4)
        if len(payload) <= char_budget:
            return payload, False

        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return payload[:char_budget].rstrip() + " [TRUNCATED]", True

        if not isinstance(data, dict):
            return payload[:char_budget].rstrip() + " [TRUNCATED]", True

        def _largest_string(obj):
            best = [0, None, None]
            def walk(container):
                if isinstance(container, dict):
                    for k, v in container.items():
                        if isinstance(v, str) and len(v) > best[0]:
                            best[0] = len(v)
                            best[1] = container
                            best[2] = k
                        walk(v)
                elif isinstance(container, list):
                    for i, v in enumerate(container):
                        if isinstance(v, str) and len(v) > best[0]:
                            best[0] = len(v)
                            best[1] = container
                            best[2] = i
                        walk(v)
            walk(obj)
            return best[1], best[2], best[0]

        truncated = False
        for _ in range(3):
            if len(json.dumps(data, ensure_ascii=False)) <= char_budget:
                break
            container, key, length = _largest_string(data)
            if container is None or length < 80:
                break
            container[key] = str(container[key])[: char_budget // 2]
            data["truncated"] = True
            truncated = True

        out = json.dumps(data, ensure_ascii=False)
        if not truncated and len(out) <= char_budget:
            return out, False
        if len(out) > char_budget:
            out = out[:char_budget].rstrip() + " [TRUNCATED]"
        return out, True