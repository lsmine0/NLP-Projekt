"""Wspólne narzędzia pomocnicze dla modułów narzędziowych agenta."""

from __future__ import annotations

from functools import wraps
import shutil
from typing import Any, Callable


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
