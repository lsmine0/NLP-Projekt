"""Wspólne narzędzia pomocnicze dla modułów narzędziowych agenta."""

from __future__ import annotations

import ast
from functools import wraps
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

_C_SYNTAX_CHECKERS: dict[str, tuple[str, str]] = {
    ".c": ("gcc", "c"),
    ".cpp": ("g++", "c++"),
    ".cc": ("g++", "c++"),
    ".cxx": ("g++", "c++"),
    ".hpp": ("g++", "c++"),
}


def validate_syntax(file_path: Path, content: str) -> None:
    """Sprawdza składnię treści dla obsługiwanych formatów (.py, .json, opcjonalnie .c/.cpp)."""
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        ast.parse(content, filename=str(file_path))
    elif suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as error:
            raise SyntaxError(
                error.msg, (str(file_path), error.lineno, error.colno, None)
            ) from error
    elif suffix in _C_SYNTAX_CHECKERS:
        compiler, lang_flag = _C_SYNTAX_CHECKERS[suffix]
        if shutil.which(compiler) is None:
            return
        try:
            proc = subprocess.run(
                [compiler, "-fsyntax-only", "-x", lang_flag, "-"],
                input=content,
                cwd=str(file_path.parent) if file_path.parent.exists() else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5.0,
            )
        except (subprocess.TimeoutExpired, OSError):
            return
        if proc.returncode != 0:
            raise SyntaxError(
                proc.stderr.strip() or f"{compiler} syntax check failed.",
                (str(file_path), None, None, None),
            )


def handle_errors(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Dekorator zwracający słownik z kluczem ``success`` i komunikatem błędu w razie wyjątku."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return function(*args, **kwargs)
        except FileNotFoundError as error:
            message = f"File or directory not found: {error.filename or error}"
        except FileExistsError as error:
            message = f"File or directory already exists: {error.filename or error}"
        except PermissionError as error:
            message = f"Permission denied: {error.filename or error}"
        except IsADirectoryError as error:
            message = f"Expected a file, but the path is a directory: {error.filename or error}"
        except NotADirectoryError as error:
            message = f"Expected a directory, but the path is not a directory: {error.filename or error}"
        except UnicodeError:
            message = "Cannot read or write the file as UTF-8 text."
        except SyntaxError as error:
            location = f" at line {error.lineno}" if error.lineno is not None else ""
            if error.offset is not None:
                location += f", column {error.offset}"
            message = f"{type(error).__name__} after edit{location}: {error.msg}. File was not modified."
        except RecursionError:
            message = "Maximum recursion depth exceeded while processing the file."
        except (OSError, ValueError, TypeError, shutil.Error) as error:
            message = f"Operation failed: {error}"

        return {"success": False, "error": message}

    return wrapper


_handle_errors = handle_errors
