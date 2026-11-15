import ast
import json
import math
import subprocess
import sys
from typing import Any, Callable
from pathlib import Path

import sympy as sp

from handlers.ToolHandle import ToolHandle


MAX_EXPRESSION_LENGTH = 512
MAX_AST_NODES = 100
MAX_POWER = 100
DEFAULT_PRECISION = 15
MAX_PRECISION = 50
CALCULATION_TIMEOUT_SECONDS = 10

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "arcsin": sp.asin,
    "acos": sp.acos,
    "arccos": sp.acos,
    "atan": sp.atan,
    "arctan": sp.atan,
    "atan2": sp.atan2,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log2": lambda value: sp.log(value, 2),
    "log10": lambda value: sp.log(value, 10),
    "abs": sp.Abs,
    "sign": sp.sign,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "factorial": sp.factorial,
    "gamma": sp.gamma,
    "binomial": sp.binomial,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "sinc": sp.sinc,
    "min": sp.Min,
    "max": sp.Max,
}
_CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
    "tau": 2 * sp.pi,
    "I": sp.I,
}
_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: sp.floor(left / right),
    ast.Mod: lambda left, right: sp.Mod(left, right),
    ast.Pow: lambda left, right: sp.Pow(left, right),
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


class _MathToolError(ValueError):
    pass


def _validate_variable(variable: str) -> sp.Symbol:
    if not isinstance(variable, str) or not variable.isidentifier() or variable.startswith("__"):
        raise _MathToolError("Zmienna musi być poprawną nazwą, np. 'x'.")
    if variable in _FUNCTIONS or variable in _CONSTANTS:
        raise _MathToolError(f"Nazwa '{variable}' jest zarezerwowana.")
    return sp.Symbol(variable)


def _validate_precision(precision: int) -> int:
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise _MathToolError("Parametr precision musi być liczbą całkowitą.")
    if not 1 <= precision <= MAX_PRECISION:
        raise _MathToolError(f"Parametr precision musi być z zakresu 1–{MAX_PRECISION}.")
    return precision


def _parse_expression(expression: str, allow_symbols: bool) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise _MathToolError("Wyrażenie nie może być puste.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise _MathToolError(
            f"Wyrażenie jest za długie (maksymalnie {MAX_EXPRESSION_LENGTH} znaków)."
        )

    try:
        parsed = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise _MathToolError(f"Błędna składnia wyrażenia: {error.msg}.") from error

    if sum(1 for _ in ast.walk(parsed)) > MAX_AST_NODES:
        raise _MathToolError(f"Wyrażenie jest zbyt złożone (maksymalnie {MAX_AST_NODES} elementów).")

    return parsed


def _to_sympy(node: ast.AST, allow_symbols: bool) -> Any:
    if isinstance(node, ast.Expression):
        return _to_sympy(node.body, allow_symbols)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _MathToolError("Dozwolone są wyłącznie literały liczbowe.")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _MathToolError("Liczba musi być skończona.")
        return sp.Rational(str(node.value))

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if node.id in _FUNCTIONS:
            raise _MathToolError(f"Funkcja '{node.id}' wymaga wywołania, np. {node.id}(x).")
        if allow_symbols and not node.id.startswith("__") and len(node.id) <= 32:
            return sp.Symbol(node.id)
        raise _MathToolError(f"Nieznana nazwa '{node.id}'.")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_to_sympy(node.operand, allow_symbols))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _to_sympy(node.left, allow_symbols)
        right = _to_sympy(node.right, allow_symbols)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise _MathToolError("Dzielenie przez zero nie jest dozwolone.")
        if isinstance(node.op, ast.Pow):
            if not right.is_number or right.is_real is not True:
                raise _MathToolError("Wykładnik potęgowania musi być skończoną liczbą rzeczywistą.")
            if abs(float(sp.N(right))) > MAX_POWER:
                raise _MathToolError(f"Wykładnik potęgowania nie może przekraczać {MAX_POWER}.")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise _MathToolError(f"Funkcja '{node.func.id}' nie jest dozwolona.")
        if node.keywords:
            raise _MathToolError("Argumenty nazwane nie są obsługiwane.")
        arguments = [_to_sympy(argument, allow_symbols) for argument in node.args]
        try:
            return function(*arguments)
        except TypeError as error:
            raise _MathToolError(f"Nieprawidłowa liczba argumentów funkcji '{node.func.id}'.") from error

    raise _MathToolError("Wyrażenie zawiera niedozwoloną składnię lub operator.")


def _failure(error: Exception) -> dict[str, Any]:
    return {"success": False, "error": str(error)}


def _calculate_expression(
    expression: str,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Any]:
    """Oblicza dozwolone wyrażenie i zwraca dokładny wynik oraz przybliżenie."""
    try:
        precision = _validate_precision(precision)
        parsed = _parse_expression(expression, allow_symbols=False)
        result = sp.simplify(_to_sympy(parsed, allow_symbols=False))
        if result.is_finite is False:
            raise _MathToolError("Wynik nie jest skończoną liczbą.")

        return {
            "success": True,
            "expression": expression,
            "result": str(result),
            "approximation": str(sp.N(result, precision)),
        }
    except _MathToolError as error:
        return _failure(error)


def _solve_equation(
    equation: str,
    variable: str = "x",
) -> dict[str, Any]:
    """Rozwiązuje równanie względem podanej zmiennej w dziedzinie liczb zespolonych."""
    if not isinstance(equation, str) or equation.count("=") != 1:
        return _failure(_MathToolError("Równanie musi zawierać dokładnie jeden znak '='."))
    if len(equation) > MAX_EXPRESSION_LENGTH:
        return _failure(
            _MathToolError(
                f"Równanie jest za długie (maksymalnie {MAX_EXPRESSION_LENGTH} znaków)."
            )
        )

    left_text, right_text = (part.strip() for part in equation.split("=", 1))
    if not left_text.strip() or not right_text.strip():
        return _failure(_MathToolError("Obie strony równania muszą zawierać wyrażenie."))

    try:
        symbol = _validate_variable(variable)
        left = _to_sympy(_parse_expression(left_text, allow_symbols=True), allow_symbols=True)
        right = _to_sympy(_parse_expression(right_text, allow_symbols=True), allow_symbols=True)
        difference = sp.simplify(left - right)

        if difference == 0:
            solution_set = sp.S.Complexes
        elif symbol not in difference.free_symbols:
            solution_set = sp.S.EmptySet
        else:
            solution_set = sp.solveset(difference, symbol, domain=sp.S.Complexes)

        if solution_set == sp.S.EmptySet:
            solutions = []
        elif solution_set == sp.S.Complexes:
            solutions = [f"wszystkie wartości {variable}"]
        elif isinstance(solution_set, sp.FiniteSet):
            solutions = [str(solution) for solution in solution_set]
        else:
            solutions = [str(solution_set)]

        return {
            "success": True,
            "equation": equation,
            "variable": variable,
            "solutions": solutions,
            "message": "Brak rozwiązań." if not solutions else None,
        }
    except (_MathToolError, NotImplementedError, ValueError) as error:
        return _failure(error)


def _differentiate_expression(
    expression: str,
    variable: str = "x",
    order: int = 1,
) -> dict[str, Any]:
    try:
        symbol = _validate_variable(variable)
        if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 10:
            raise _MathToolError("Rząd pochodnej musi być liczbą całkowitą z zakresu 1–10.")
        parsed = _parse_expression(expression, allow_symbols=True)
        result = sp.diff(_to_sympy(parsed, allow_symbols=True), symbol, order)
        return {
            "success": True,
            "expression": expression,
            "variable": variable,
            "order": order,
            "result": str(result),
        }
    except (_MathToolError, NotImplementedError, ValueError) as error:
        return _failure(error)


def _integrate_expression(
    expression: str,
    variable: str = "x",
    lower_bound: str | None = None,
    upper_bound: str | None = None,
) -> dict[str, Any]:
    try:
        symbol = _validate_variable(variable)
        if (lower_bound is None) != (upper_bound is None):
            raise _MathToolError("Całka oznaczona wymaga podania obu granic.")
        parsed = _parse_expression(expression, allow_symbols=True)
        integrand = _to_sympy(parsed, allow_symbols=True)
        if lower_bound is None:
            result = sp.integrate(integrand, symbol)
        else:
            lower = _to_sympy(_parse_expression(lower_bound, allow_symbols=True), allow_symbols=True)
            upper = _to_sympy(_parse_expression(upper_bound, allow_symbols=True), allow_symbols=True)
            result = sp.integrate(integrand, (symbol, lower, upper))
        return {
            "success": True,
            "expression": expression,
            "variable": variable,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "result": str(result),
        }
    except (_MathToolError, NotImplementedError, ValueError) as error:
        return _failure(error)


def _calculate_limit(
    expression: str,
    variable: str = "x",
    point: str = "0",
    direction: str = "+-",
) -> dict[str, Any]:
    try:
        symbol = _validate_variable(variable)
        if direction not in ("+", "-", "+-"):
            raise _MathToolError("Kierunek granicy musi mieć wartość '+', '-' albo '+-'.")
        parsed_expression = _parse_expression(expression, allow_symbols=True)
        parsed_point = _parse_expression(point, allow_symbols=True)
        result = sp.limit(
            _to_sympy(parsed_expression, allow_symbols=True),
            symbol,
            _to_sympy(parsed_point, allow_symbols=True),
            dir=direction,
        )
        return {
            "success": True,
            "expression": expression,
            "variable": variable,
            "point": point,
            "direction": direction,
            "result": str(result),
        }
    except (_MathToolError, NotImplementedError, ValueError) as error:
        return _failure(error)


def _execute_math_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    if operation == "calculate_expression":
        return _calculate_expression(
            request.get("expression"),
            request.get("precision", DEFAULT_PRECISION),
        )
    if operation == "solve_equation":
        return _solve_equation(
            request.get("equation"),
            request.get("variable", "x"),
        )
    if operation == "differentiate_expression":
        return _differentiate_expression(
            request.get("expression"),
            request.get("variable", "x"),
            request.get("order", 1),
        )
    if operation == "integrate_expression":
        return _integrate_expression(
            request.get("expression"),
            request.get("variable", "x"),
            request.get("lower_bound"),
            request.get("upper_bound"),
        )
    if operation == "calculate_limit":
        return _calculate_limit(
            request.get("expression"),
            request.get("variable", "x"),
            request.get("point", "0"),
            request.get("direction", "+-"),
        )
    return _failure(_MathToolError("Nieznana operacja matematyczna."))


def _run_isolated(request: dict[str, Any]) -> dict[str, Any]:
    worker_code = (
        "import json, sys; "
        "from tools.MathTools import _execute_math_request; "
        "json.dump(_execute_math_request(json.load(sys.stdin)), sys.stdout, ensure_ascii=True)"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", worker_code],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=CALCULATION_TIMEOUT_SECONDS,
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failure(
            _MathToolError(
                f"Przekroczono limit czasu obliczeń ({CALCULATION_TIMEOUT_SECONDS} s)."
            )
        )
    except OSError as error:
        return _failure(_MathToolError(f"Nie udało się uruchomić procesu obliczeniowego: {error}"))

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"kod zakończenia {completed.returncode}"
        return _failure(_MathToolError(f"Proces obliczeniowy zakończył się błędem: {detail}"))
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return _failure(_MathToolError(f"Nieprawidłowa odpowiedź procesu obliczeniowego: {error}"))
    if not isinstance(result, dict):
        return _failure(_MathToolError("Proces obliczeniowy zwrócił nieprawidłowy wynik."))
    return result


def calculate_expression(
    expression: str,
    precision: int = DEFAULT_PRECISION,
) -> dict[str, Any]:
    """Oblicza wyrażenie w odizolowanym procesie z limitem czasu."""
    return _run_isolated(
        {"operation": "calculate_expression", "expression": expression, "precision": precision}
    )


def solve_equation(
    equation: str,
    variable: str = "x",
) -> dict[str, Any]:
    """Rozwiązuje równanie w odizolowanym procesie z limitem czasu."""
    return _run_isolated(
        {"operation": "solve_equation", "equation": equation, "variable": variable}
    )


def differentiate_expression(
    expression: str,
    variable: str = "x",
    order: int = 1,
) -> dict[str, Any]:
    """Oblicza pochodną w odizolowanym procesie z limitem czasu."""
    return _run_isolated(
        {
            "operation": "differentiate_expression",
            "expression": expression,
            "variable": variable,
            "order": order,
        }
    )


def integrate_expression(
    expression: str,
    variable: str = "x",
    lower_bound: str | None = None,
    upper_bound: str | None = None,
) -> dict[str, Any]:
    """Oblicza całkę nieoznaczoną lub oznaczoną w odizolowanym procesie."""
    return _run_isolated(
        {
            "operation": "integrate_expression",
            "expression": expression,
            "variable": variable,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    )


def calculate_limit(
    expression: str,
    variable: str = "x",
    point: str = "0",
    direction: str = "+-",
) -> dict[str, Any]:
    """Oblicza granicę w odizolowanym procesie z limitem czasu."""
    return _run_isolated(
        {
            "operation": "calculate_limit",
            "expression": expression,
            "variable": variable,
            "point": point,
            "direction": direction,
        }
    )


def register_math_tools(tool_handle: ToolHandle, overwrite: bool = True) -> None:
    """Rejestruje definicje narzędzi matematycznych w obiekcie ToolHandle."""
    tool_handle.create_tool(
        name="calculate_expression",
        description=(
            "Oblicza dozwolone wyrażenie bez wykonywania kodu. Obsługuje podstawową arytmetykę, "
            "potęgowanie, stałe pi/e/tau/I oraz funkcje trygonometryczne i odwrotne, hiperboliczne, "
            "logarytmy, sqrt, abs, sign, floor/ceil, factorial, gamma, binomial, erf, erfc, sinc, "
            "min i max. Zwraca wynik dokładny oraz przybliżenie. Nieznane funkcje są odrzucane. "
            f"Limit czasu: {CALCULATION_TIMEOUT_SECONDS} sekund."
        ),
        properties={
            "expression": {
                "type": "string",
                "description": "Wyrażenie matematyczne, np. 'sqrt(2) + sin(pi/2)' lub '2^10'.",
            },
            "precision": {
                "type": "integer",
                "description": f"Liczba cyfr precyzji przybliżenia (1–{MAX_PRECISION}, domyślnie {DEFAULT_PRECISION}).",
            },
        },
        required=["expression"],
        overwrite=overwrite,
    )
    tool_handle.create_tool(
        name="solve_equation",
        description=(
            "Rozwiązuje równanie algebraiczne w dziedzinie liczb zespolonych. "
            "Równanie podaj jako tekst z jednym znakiem '=', np. 'x^2 - 4 = 0'. "
            f"Limit czasu jednego obliczenia wynosi {CALCULATION_TIMEOUT_SECONDS} sekund."
        ),
        properties={
            "equation": {
                "type": "string",
                "description": "Równanie, np. '2*x + 3 = 7'.",
            },
            "variable": {
                "type": "string",
                "description": "Nazwa zmiennej, względem której rozwiązać równanie (domyślnie 'x').",
            },
        },
        required=["equation"],
        overwrite=overwrite,
    )
    tool_handle.create_tool(
        name="differentiate_expression",
        description=(
            "Oblicza pochodną wyrażenia względem zmiennej. Obsługuje pochodne wyższych rzędów "
            "(rząd od 1 do 10). Przykład: expression='sin(x)^2', variable='x'. "
            f"Limit czasu: {CALCULATION_TIMEOUT_SECONDS} sekund."
        ),
        properties={
            "expression": {"type": "string", "description": "Wyrażenie, np. 'x^3 + sin(x)'."},
            "variable": {"type": "string", "description": "Zmienna, domyślnie 'x'."},
            "order": {"type": "integer", "description": "Rząd pochodnej od 1 do 10, domyślnie 1."},
        },
        required=["expression"],
        overwrite=overwrite,
    )
    tool_handle.create_tool(
        name="integrate_expression",
        description=(
            "Oblicza całkę nieoznaczoną lub oznaczoną. Dla całki oznaczonej podaj obie granice. "
            "Przykład: expression='x^2', lower_bound='0', upper_bound='1'. "
            f"Limit czasu: {CALCULATION_TIMEOUT_SECONDS} sekund."
        ),
        properties={
            "expression": {"type": "string", "description": "Całkowana funkcja."},
            "variable": {"type": "string", "description": "Zmienna całkowania, domyślnie 'x'."},
            "lower_bound": {"type": "string", "description": "Dolna granica (opcjonalna)."},
            "upper_bound": {"type": "string", "description": "Górna granica (opcjonalna)."},
        },
        required=["expression"],
        overwrite=overwrite,
    )
    tool_handle.create_tool(
        name="calculate_limit",
        description=(
            "Oblicza granicę wyrażenia. Kierunek może wynosić '+', '-' albo '+-' (dwustronnie). "
            "Przykład: expression='sin(x)/x', point='0', direction='+-'. "
            f"Limit czasu: {CALCULATION_TIMEOUT_SECONDS} sekund."
        ),
        properties={
            "expression": {"type": "string", "description": "Wyrażenie, np. 'sin(x)/x'."},
            "variable": {"type": "string", "description": "Zmienna granicy, domyślnie 'x'."},
            "point": {"type": "string", "description": "Punkt granicy, domyślnie '0'."},
            "direction": {
                "type": "string",
                "enum": ["+", "-", "+-"],
                "description": "Kierunek granicy: prawostronna, lewostronna albo dwustronna.",
            },
        },
        required=["expression"],
        overwrite=overwrite,
    )

    for name, function in AVAILABLE_MATH_TOOLS.items():
        tool_handle.register_function(name, function)


AVAILABLE_MATH_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculate_expression": calculate_expression,
    "solve_equation": solve_equation,
    "differentiate_expression": differentiate_expression,
    "integrate_expression": integrate_expression,
    "calculate_limit": calculate_limit,
}
