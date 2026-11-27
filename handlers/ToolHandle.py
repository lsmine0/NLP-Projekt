import json
from typing import Any, Dict, Iterable, List, Optional
import contextvars
import inspect
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from core import Logger
from core.ToolPolicy import (
    DEFAULT_RISK_POLICIES,
    PreparedToolCall,
    ToolSpec,
    is_protected,
    most_restrictive_policy,
    normalize_policy,
    normalize_protected_paths,
    normalize_risk,
    normalize_roots,
    normalize_tool_path,
)

# #51: zakres sesji (chat_id + attachment_store) widoczny dla narzędzi.
# ContextVar NIE propaguje się do wątków ThreadPoolExecutor (AgentFlow._ExecuteToolCalls),
# dlatego scope ustawia się WEWNĄTRZ zadania submitowanego do executora
# (AgentFlow._RunToolWithScope), a nie przed submit(). tools/ImageTools.py pyta przez
# current_tool_session() - inaczej narzędzie nie znałoby, z którego czatu pochodzi ID.
_TOOL_SESSION: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tool_session", default=None
)


# Jedno źródło skracania opisów narzędzi: schema wysyłana modelowi, widok UI
# i meta-narzędzie list_tools korzystają z tego samego helpera. Wcześniej
# PromptBuilder i OllamaHandle utrzymywali własne, rozbieżne kopie tej funkcji.
SHORT_DESCRIPTION_LIMIT = 60
# AgentFlow obcina wynik każdego narzędzia do DEFAULT_TOOL_CHARS (1024 znaków),
# a TruncateJson nie umie złożyć urwanego JSONa. Meta-narzędzie list_tools
# mieści się więc w tym limitu samo - patrz _fit_to_char_budget.
INTROSPECTION_CHAR_BUDGET = 960


def short_tool_description(value: Any, limit: int = SHORT_DESCRIPTION_LIMIT) -> str:
    """Skróć opis narzędzia do jednej linii (max ``limit`` znaków)."""
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "\u2026"
    return text


def _longest_string(obj: Any) -> tuple[Any, Any, int]:
    """Znajdź najdłuższy string w payloadu: (container, key, długość)."""
    best: list[Any] = [None, None, 0]

    def walk(container: Any) -> None:
        if isinstance(container, dict):
            items = container.items()
        elif isinstance(container, list):
            items = list(enumerate(container))
        else:
            return
        for key, value in items:
            if isinstance(value, str):
                if len(value) > best[2]:
                    best[:] = [container, key, len(value)]
            elif isinstance(value, (dict, list)):
                walk(value)

    walk(obj)
    return best[0], best[1], best[2]


def _fit_to_char_budget(payload: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """Skróć najdłuższe stringi, aż JSON payloadu mieści się w ``budget``.

    Zawsze zwraca poprawny JSON: urwanego JSONa model nie umiałby przeczytać,
    a AgentFlow docina rezultaty narzędzi hard-cutem z "[TRUNCATED]".
    """
    for _ in range(12):
        serialized = len(json.dumps(payload, ensure_ascii=False))
        if serialized <= budget:
            break
        container, key, length = _longest_string(payload)
        if container is None or length <= 16:
            break
        keep = max(8, length - (serialized - budget) - 1)  # 1 znak na "…"
        if keep >= length:
            break
        text = str(container[key])
        container[key] = text[:keep].rstrip() + "\u2026"
    return payload


def current_tool_session() -> Optional[dict]:
    """Zakres bieżącej sesji: {'chat_id': ..., 'attachment_store': ...} albo None."""
    return _TOOL_SESSION.get()


@contextmanager
def use_tool_session(scope):
    """Ustawia zakres narzędzi w wątku wykonującym tool i sprząta po nim (#51)."""
    token = _TOOL_SESSION.set(dict(scope) if isinstance(scope, dict) else {})
    try:
        yield _TOOL_SESSION.get()
    finally:
        _TOOL_SESSION.reset(token)


# Zgoda wydana przez politykę rejestru (allow lub decyzję użytkownika po
# zdarzeniu approval_required) musi być widoczna dla samego narzędzia, inaczej
# run_python odmówiłby wykonania po tym, jak użytkownik już zezwolił. ContextVar
# nie propaguje się do wątków executora, dlatego scope ustawia się wewnątrz
# execute_prepared - to ten wątek faktycznie wykonuje funkcję narzędzia.
_TOOL_AUTHORITY: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tool_authority", default=None
)


def current_tool_authority() -> Optional[dict]:
    """Zgoda hosta dla bieżącego wywołania: {'tool':..., 'policy':...} albo None."""
    return _TOOL_AUTHORITY.get()


@contextmanager
def use_tool_authority(scope):
    """Ustawia zgodę hosta w wątku narzędzia i sprząta po niej."""
    token = _TOOL_AUTHORITY.set(dict(scope) if isinstance(scope, dict) else {})
    try:
        yield _TOOL_AUTHORITY.get()
    finally:
        _TOOL_AUTHORITY.reset(token)


class ToolErrorText(str):
    """Tekst błędu dla narzędzi, których bezpośrednie API zwraca str."""


class ToolHandle:
    """
    Klasa realizująca operacje CRUD na słowniku narzędzi (tools)
    zgodnie ze specyfikacją Ollama Tool Calling.
    """

    def __init__(
        self,
        *,
        allowed_directories: Iterable[str | Path] | None = None,
        primary_root: str | Path | None = None,
        policy_overrides: Dict[str, str] | None = None,
        risk_policies: Dict[str, str] | None = None,
        protected_paths: Iterable[str | Path] | None = None,
    ) -> None:
        # Słownik przechowujący narzędzia (klucz: nazwa narzędzia)
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._functions: Dict[str, Any] = {}
        self._specs: Dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        self._allowed_roots: tuple[Path, ...] = ()
        self._primary_root: Path | None = None
        self._protected_paths: tuple[Path, ...] = ()
        self._policy_overrides: Dict[str, str] = {}
        self._risk_policies: Dict[str, str] = dict(DEFAULT_RISK_POLICIES)
        self.configure_security(
            allowed_directories=allowed_directories,
            primary_root=primary_root,
            policy_overrides=policy_overrides,
            risk_policies=risk_policies,
            protected_paths=protected_paths,
        )

    # ------------------------------------------------------------------
    # Security / policy configuration
    # ------------------------------------------------------------------
    def configure_security(
        self,
        *,
        allowed_directories: Iterable[str | Path] | None = None,
        primary_root: str | Path | None = None,
        policy_overrides: Dict[str, str] | None = None,
        risk_policies: Dict[str, str] | None = None,
        protected_paths: Iterable[str | Path] | None = None,
    ) -> None:
        with self._lock:
            if allowed_directories is not None:
                roots = normalize_roots(allowed_directories)
                self._allowed_roots = roots
                if primary_root is not None:
                    resolved_primary = Path(primary_root).expanduser().resolve(strict=False)
                    self._primary_root = resolved_primary
                    if roots and resolved_primary not in roots:
                        self._allowed_roots = (resolved_primary, *roots)
                else:
                    self._primary_root = roots[0] if roots else None
            elif primary_root is not None:
                self._primary_root = Path(primary_root).expanduser().resolve(strict=False)

            if protected_paths is not None:
                self._protected_paths = normalize_protected_paths(protected_paths)
            if policy_overrides is not None:
                self._policy_overrides = self._normalize_policy_map(policy_overrides)
            if risk_policies is not None:
                self._risk_policies = {
                    **DEFAULT_RISK_POLICIES,
                    **self._normalize_policy_map(risk_policies),
                }

    @staticmethod
    def _normalize_policy_map(values: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(values, dict):
            raise ValueError("Tool policy configuration must be a dictionary.")
        normalized: Dict[str, str] = {}
        for key, value in values.items():
            policy = normalize_policy(value)
            if policy is not None:
                normalized[str(key)] = policy
        return normalized

    def set_policy_config(
        self,
        *,
        policy_overrides: Dict[str, str] | None = None,
        risk_policies: Dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            if policy_overrides is not None:
                self._policy_overrides = self._normalize_policy_map(policy_overrides)
            if risk_policies is not None:
                self._risk_policies = {
                    **DEFAULT_RISK_POLICIES,
                    **self._normalize_policy_map(risk_policies),
                }

    def get_policy(self, name: str, spec: ToolSpec | None = None) -> str:
        with self._lock:
            spec = spec or self._specs.get(name)
            if name in self._policy_overrides:
                return self._policy_overrides[name]
            if spec is not None:
                # A policy registered with the tool (e.g. MCP tools.ask) is more
                # specific than risk-label defaults, but less specific than an
                # explicit per-tool override from settings.
                if spec.policy:
                    return spec.policy
                risk_policies = [
                    self._risk_policies[label]
                    for label in spec.risk
                    if label in self._risk_policies
                ]
                resolved = most_restrictive_policy(risk_policies)
                if resolved:
                    return resolved
            return "allow"

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------
    def create_tool(
        self,
        name: str,
        description: str,
        properties: Optional[Dict[str, Dict[str, Any]]] = None,
        required: Optional[List[str]] = None,
        function=None,
        overwrite: bool = False,
        *,
        risk: str | Iterable[str] | None = None,
        policy: str | None = None,
        source: str = "builtin",
        server: str | None = None,
        remote_name: str | None = None,
        path_args: Iterable[str] | str | None = None,
        kind: str = "tool",
    ) -> Dict[str, Any]:
        """Tworzy nowe narzędzie i dodaje je do kolekcji."""
        if name in self._tools and not overwrite:
            raise ValueError(f"Narzędzie '{name}' już istnieje.")

        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "required": required or [],
                    "properties": properties or {},
                },
            },
        }

        with self._lock:
            self._tools[name] = tool_schema
            self._specs[name] = ToolSpec(
                name=name,
                risk=normalize_risk(risk),
                policy=normalize_policy(policy),
                source=source,
                server=server,
                remote_name=remote_name,
                path_args=tuple([path_args] if isinstance(path_args, str) else (path_args or ())),
                kind=kind,
                description=description,
            )

        if function is not None:
            self.register_function(name, function)

        return tool_schema

    def create_tool_from_schema(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function=None,
        overwrite: bool = False,
        *,
        risk: str | Iterable[str] | None = None,
        policy: str | None = None,
        source: str = "builtin",
        server: str | None = None,
        remote_name: str | None = None,
        path_args: Iterable[str] | str | None = None,
        kind: str = "tool",
    ) -> Dict[str, Any]:
        """Tworzy narzędzie z gotowym schematem parametrów (np. z MCP).

        W odróżnieniu od ``create_tool`` nie buduje schematu od zera, więc
        zachowuje pełną strukturę JSON Schema dostarczoną przez serwer MCP.
        """
        if name in self._tools and not overwrite:
            raise ValueError(f"Narzędzie '{name}' już istnieje.")

        if not isinstance(parameters, dict):
            raise TypeError("Tool parameters must be a dictionary.")

        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        with self._lock:
            self._tools[name] = tool_schema
            self._specs[name] = ToolSpec(
                name=name,
                risk=normalize_risk(risk),
                policy=normalize_policy(policy),
                source=source,
                server=server,
                remote_name=remote_name,
                path_args=tuple([path_args] if isinstance(path_args, str) else (path_args or ())),
                kind=kind,
                description=description,
            )

        if function is not None:
            self.register_function(name, function)

        return tool_schema

    def register_function(self, name: str, function) -> None:

        if name not in self._tools:
            raise KeyError(f"Tool '{name}' does not exist.")

        if not callable(function):
            raise TypeError("Expected a callable Python function.")

        with self._lock:
            self._functions[name] = function

    def add_parameter(
        self,
        tool_name: str,
        param_name: str,
        param_type: str,
        description: str,
        required: bool = True,
    ) -> Dict[str, Any]:
        """Dodaje pojedynczy parametr do istniejącego narzędzia."""
        if tool_name not in self._tools:
            raise KeyError(f"Narzędzie '{tool_name}' nie istnieje.")

        params = self._tools[tool_name]["function"]["parameters"]
        params["properties"][param_name] = {
            "type": param_type,
            "description": description,
        }

        if required and param_name not in params["required"]:
            params["required"].append(param_name)

        return self._tools[tool_name]

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Pobiera pojedyncze narzędzie po nazwie."""
        return self._tools.get(name)

    def get_tool_spec(self, name: str) -> Optional[ToolSpec]:
        with self._lock:
            return self._specs.get(name)

    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Zwraca listę narzędzi: [ {...}, {...} ]."""
        return list(self._tools.values())

    def get_tools_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Zwraca pełny słownik: {"tools": [ {...} ]}."""
        return {"tools": self.get_tools_list()}

    # ------------------------------------------------------------------
    # Introspekcja (meta-narzędzie list_tools, UI, testy)
    # ------------------------------------------------------------------
    def tool_names(self) -> List[str]:
        """Nazwy zarejestrowanych narzędzi w kolejności rejestracji."""
        with self._lock:
            return list(self._tools)

    def tool_summaries(self) -> List[Dict[str, str]]:
        """Kompaktowa lista: [{'name': ..., 'description': <max 60 znaków>}].

        To jedyny format, z którego ktokolwiek (prompt, schema, UI) buduje
        „przegląd” narzędzi - model nie musi już dublować opisów w prompcie.
        """
        with self._lock:
            items = [(name, self._tools[name]) for name in self._tools]
        summaries: List[Dict[str, str]] = []
        for name, schema in items:
            description = (schema.get("function") or {}).get("description", "")
            summaries.append(
                {"name": name, "description": short_tool_description(description)}
            )
        return summaries

    def describe_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Zwraca unifikowany, pełny opis jednego narzędzia albo None."""
        with self._lock:
            schema = self._tools.get(name)
            spec = self._specs.get(name)
        if schema is None and spec is None:
            return None
        function = (schema or {}).get("function") or {}
        record: Dict[str, Any] = (
            spec.public(self.get_policy(name, spec))
            if spec is not None
            else {"name": name}
        )
        record["description"] = function.get("description") or record.get("description", "")
        record["parameters"] = function.get("parameters") or {}
        return record

    @staticmethod
    def _tool_view(record: Dict[str, Any]) -> Dict[str, Any]:
        """Zwarty widok narzędzia dla list_tools: typy + opisy, bez szumu.

        Raw JSON Schema potrafi zajmować więcej niż limit rezultatu narzędzia
        (1024 znaków w AgentFlow), więc renderujemy tylko to, czego model
        potrzebuje do wywołania: wymagane pola, typy i krótkie opisy.
        """
        parameters = record.get("parameters") or {}
        properties: Dict[str, Dict[str, Any]] = {}
        for param_name, schema in (parameters.get("properties") or {}).items():
            if not isinstance(schema, dict):
                continue
            entry: Dict[str, Any] = {"type": schema.get("type", "string")}
            items = schema.get("items")
            if isinstance(items, dict) and items.get("type"):
                entry["items"] = {"type": items["type"]}
            allowed = schema.get("enum")
            if isinstance(allowed, list) and len(json.dumps(allowed)) <= 60:
                entry["enum"] = allowed
            description = short_tool_description(schema.get("description", ""), 90)
            if description:
                entry["description"] = description
            properties[str(param_name)] = entry

        view: Dict[str, Any] = {
            "name": record.get("name"),
            "description": short_tool_description(record.get("description", ""), 180),
            "required": [
                item for item in (parameters.get("required") or []) if isinstance(item, str)
            ],
            "parameters": properties,
            "risk": record.get("risk", []),
            "policy": record.get("policy", "allow"),
            "source": record.get("source", "builtin"),
        }
        if record.get("server"):
            view["server"] = record["server"]
        return view

    def list_available_tools(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Meta-narzędzie: agent sam zagląda do swojego rejestru narzędzi.

        Bez argumentów zwraca listę nazw wszystkich dostępnych narzędzi (krótkie
        opisy model już widzi w listie ``tools``, więc nie dublujemy ich).
        Z ``tool_name`` zwraca pełny opis i parametry (typy, wymagane pola,
        opisy parametrów) jednego narzędzia - dokładnie to, czego nie ma w
        kompaktowej listie ``tools`` wysyłanej z żądaniem.

        Kształt odpowiedzi trzyma konwencję rejestru: ``success``/``error`` są
        na górnym poziomie, a reszta ląduje w ``result`` (ToolHandle._format_result),
        dlatego payload nie może być własnym kluczem "result".
        """
        if tool_name is not None:
            if not isinstance(tool_name, str) or not tool_name.strip():
                return {
                    "success": False,
                    "error": "tool_name musi być niepustym tekstem (nazwą narzędzia).",
                }
            wanted = tool_name.strip()
            record = self.describe_tool(wanted)
            if record is None:
                known = ", ".join(self.tool_names())
                return {
                    "success": False,
                    "error": (
                        f"Narzędzie '{wanted}' nie jest zarejestrowane. "
                        f"Dostępne: {known}."
                    ),
                }
            view = self._tool_view(record)
            return {
                "success": True,
                "tool": _fit_to_char_budget(view, INTROSPECTION_CHAR_BUDGET),
            }
        # Przegląd zawiera tylko nazwy: krótkie opisy model już widzi w listie
        # `tools`, więc powtarzać je w wyniku narzędzia = dublować kontekst.
        return {
            "success": True,
            "count": len(self.tool_names()),
            "tools": self.tool_names(),
            "hint": (
                "list_tools(tool_name=\"<nazwa>\") zwraca pełny opis i parametry "
                "jednego narzędzia."
            ),
        }

    def public_tools(self) -> List[Dict[str, Any]]:
        with self._lock:
            names = list(self._specs)
            specs = [self._specs[name] for name in names]
        return [spec.public(self.get_policy(spec.name, spec)) for spec in specs]

    def to_json(self, indent: int = 2) -> str:
        """Zwraca słownik narzędzi sformatowany jako tekst JSON."""
        return json.dumps(self.get_tools_dict(), indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    def update_tool(
        self,
        name: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Dict[str, Any]]] = None,
        required: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Aktualizuje wybrane pola istniejącego narzędzia."""
        if name not in self._tools:
            raise KeyError(f"Narzędzie '{name}' nie istnieje.")

        func_data = self._tools[name]["function"]

        if description is not None:
            func_data["description"] = description
            with self._lock:
                spec = self._specs.get(name)
                if spec is not None:
                    self._specs[name] = replace(spec, description=description)
        if properties is not None:
            func_data["parameters"]["properties"] = properties
        if required is not None:
            func_data["parameters"]["required"] = required

        return self._tools[name]

    def remove_parameter(self, tool_name: str, param_name: str) -> Dict[str, Any]:
        """Usuwa pojedynczy parametr z narzędzia."""
        if tool_name not in self._tools:
            raise KeyError(f"Narzędzie '{tool_name}' nie istnieje.")

        params = self._tools[tool_name]["function"]["parameters"]
        params["properties"].pop(param_name, None)
        if param_name in params["required"]:
            params["required"].remove(param_name)

        return self._tools[tool_name]

    # ------------------------------------------------------------------
    # Execution / policy gate
    # ------------------------------------------------------------------
    def prepare_tool_call(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> PreparedToolCall:
        """Validate arguments/paths and resolve the effective policy.

        This method never executes the tool. It returns a structured denial for
        unknown tools, bad arguments, paths outside allowed roots or a ``deny``
        policy. ``ask`` is reported through ``requires_approval`` so the caller
        can request a GUI/CLI decision before execution.
        """

        with self._lock:
            function = self._functions.get(name)
            spec = self._specs.get(name)
        if spec is None:
            spec = ToolSpec(name=name, description="")
        if function is None:
            return PreparedToolCall(
                name=name,
                arguments={},
                spec=spec,
                policy="deny",
                requires_approval=False,
                denial={
                    "code": "unknown_tool",
                    "message": f"Narzędzie '{name}' nie jest zarejestrowane.",
                },
            )
        if not isinstance(arguments, dict):
            return PreparedToolCall(
                name=name,
                arguments={},
                spec=spec,
                policy="deny",
                denial={
                    "code": "invalid_arguments",
                    "message": "Argumenty narzędzia muszą być obiektem.",
                },
            )

        normalized = dict(arguments)
        signature = inspect.signature(function)
        has_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not has_var_kwargs:
            accepted = {
                parameter_name
                for parameter_name, parameter in signature.parameters.items()
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            unknown = [key for key in normalized if key not in accepted]
            if unknown:
                # Local models sometimes add harmless extra fields (e.g. region for
                # search_web). Dropping them is safer than losing the whole tool call.
                Logger.printWarning(
                    f"Narzędzie '{name}': pomijam nieznane argumenty: "
                    f"{', '.join(sorted(unknown))}."
                )
                for key in unknown:
                    normalized.pop(key, None)
        try:
            signature.bind(**normalized)
        except TypeError as error:
            return PreparedToolCall(
                name=name,
                arguments=normalized,
                spec=spec,
                policy="deny",
                denial={
                    "code": "invalid_arguments",
                    "message": f"Nieprawidłowe argumenty narzędzia: {error}",
                },
            )

        if self._allowed_roots and spec.path_args:
            for key in spec.path_args:
                if key not in normalized:
                    continue
                try:
                    resolved = normalize_tool_path(
                        normalized[key],
                        allowed_roots=self._allowed_roots,
                        primary_root=self._primary_root,
                    )
                except (ValueError, PermissionError, OSError, RuntimeError) as error:
                    return PreparedToolCall(
                        name=name,
                        arguments=normalized,
                        spec=spec,
                        policy="deny",
                        denial={
                            "code": "path_outside_allowed",
                            "message": str(error),
                        },
                    )
                if is_protected(resolved, self._protected_paths):
                    return PreparedToolCall(
                        name=name,
                        arguments=normalized,
                        spec=spec,
                        policy="deny",
                        denial={
                            "code": "protected_path",
                            "message": "Ścieżka jest chroniona i nie może być modyfikowana.",
                        },
                    )
                normalized[key] = str(resolved)

        policy = self.get_policy(name, spec)
        if policy == "deny":
            return PreparedToolCall(
                name=name,
                arguments=normalized,
                spec=spec,
                policy=policy,
                denial={
                    "code": "policy_denied",
                    "message": "Narzędzie jest zablokowane przez politykę allow/ask/deny.",
                },
            )

        return PreparedToolCall(
            name=name,
            arguments=normalized,
            spec=spec,
            policy=policy,
            requires_approval=policy == "ask",
        )

    @staticmethod
    def denial_payload(
        prepared: PreparedToolCall | ToolSpec | str,
        code: str,
        message: str,
        *,
        approval_id: str | None = None,
    ) -> Dict[str, Any]:
        if isinstance(prepared, PreparedToolCall):
            name = prepared.name
            risk = prepared.risk
        elif isinstance(prepared, ToolSpec):
            name = prepared.name
            risk = prepared.risk
        else:
            name = str(prepared)
            risk = frozenset()
        result: Dict[str, Any] = {
            "denied": True,
            "code": code,
            "tool": name,
            "risk": list(risk),
        }
        if approval_id:
            result["approval_id"] = approval_id
        return {"success": False, "error": message, "result": result}

    def denial_json(
        self,
        prepared: PreparedToolCall | ToolSpec | str,
        code: str,
        message: str,
        *,
        approval_id: str | None = None,
    ) -> str:
        return json.dumps(
            self.denial_payload(prepared, code, message, approval_id=approval_id),
            ensure_ascii=False,
        )

    def execute_prepared(self, prepared: PreparedToolCall) -> str:
        """Execute a call that already passed validation/policy/approval."""

        if prepared.denial is not None:
            return self.denial_json(
                prepared,
                str(prepared.denial.get("code", "denied")),
                str(prepared.denial.get("message", "Narzędzie nie zostało wykonane.")),
            )

        try:
            with self._lock:
                function = self._functions.get(prepared.name)
            if function is None:
                raise KeyError(f"No implementation for tool '{prepared.name}'.")
            # Wywołanie przeszło walidację i politykę (opcjonalnie także zgoda
            # użytkownika) - narzędzia mogą to sprawdzić przez
            # handlers.ToolHandle.current_tool_authority().
            with use_tool_authority({
                "tool": prepared.name,
                "policy": prepared.policy,
            }):
                result = function(**prepared.arguments)
            return self._format_result(result)
        except Exception as error:
            return json.dumps(
                {"success": False, "error": str(error), "result": None},
                ensure_ascii=False,
            )

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, ToolErrorText):
            output = {"success": False, "error": str(result), "result": None}
        elif isinstance(result, dict) and isinstance(result.get("success"), bool):
            success = result["success"]
            payload = {key: value for key, value in result.items() if key not in ("success", "error")}
            output = {
                "success": success,
                "error": None if success else result.get("error"),
                "result": payload if success else None,
            }
        else:
            output = {"success": True, "error": None, "result": result}
        return json.dumps(output, ensure_ascii=False, default=str)

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Wykonuje narzędzie i zwraca JSON z polami success, error i result.

        Calls requiring approval are denied with ``approval_required``; the agent
        loop must request a decision and call ``execute_prepared`` itself.
        """
        prepared = self.prepare_tool_call(name, arguments)
        if prepared.denial is not None:
            return self.denial_json(
                prepared,
                str(prepared.denial.get("code", "denied")),
                str(prepared.denial.get("message", "Narzędzie nie zostało wykonane.")),
            )
        if prepared.requires_approval:
            return self.denial_json(
                prepared,
                "approval_required",
                "Narzędzie wymaga zgody użytkownika.",
            )
        return self.execute_prepared(prepared)

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------
    def delete_tool(self, name: str) -> bool:
        """Usuwa narzędzie. Zwraca True, jeśli narzędzie istniało i zostało usunięte."""
        with self._lock:
            schema = self._tools.pop(name, None)
            self._functions.pop(name, None)
            self._specs.pop(name, None)
        return schema is not None

    def clear_tools(self) -> None:
        """Usuwa wszystkie narzędzia."""
        with self._lock:
            self._tools.clear()
            self._functions.clear()
            self._specs.clear()
