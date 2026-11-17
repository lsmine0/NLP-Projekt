"""Narzędzia kodowania dla agenta: precyzyjna edycja plików i kontrolowane uruchamianie kodu.

UWAGA BEZPIECZEŃSTWA: Moduł ogranicza ścieżki do ``workspace_dir`` i uruchamia kod
w osobnym podprocesie z ograniczonym ``cwd`` (dla Pythona dodatkowo z flagą ``-I``).
Nie stanowi to pełnej izolacji systemowej (sandboxa OS), dlatego uruchamianie kodu
wymaga jawnej zgody użytkownika po stronie hosta.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable

from core import Logger
from handlers.ToolHandle import ToolHandle
from tools.FileTools import write_file
from tools.ToolUtils import handle_errors as _handle_errors, validate_syntax

DEFAULT_TIMEOUT: float = 10.0
MAX_TIMEOUT: float = 60.0
MAX_OUTPUT_CHARS: int = 10_000
ISOLATION_WARNING: str = (
    "UWAGA: Uruchamianie kodu odbywa się w podprocesie z ograniczonym katalogiem "
    "roboczym (cwd) i flagą -I, co NIE stanowi pełnej izolacji systemowej (sandboxa)."
)

_default_workspace_dir: Path | None = None
_default_allow_execution: bool = False
_default_approval_callback: Callable[[str], bool] | None = None


def configure_code_tools(
    *,
    workspace_dir: str | Path | None = None,
    allow_execution: bool = False,
    approval_callback: Callable[[str], bool] | None = None,
) -> None:
    """Konfiguruje domyślny katalog roboczy i politykę zgody po stronie hosta."""
    global _default_workspace_dir, _default_allow_execution, _default_approval_callback
    _default_workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
    _default_allow_execution = bool(allow_execution)
    _default_approval_callback = approval_callback


def _has_symlink_component(path: Path, *, stop_at: Path | None = None) -> bool:
    current = path
    while True:
        if stop_at is not None and current == stop_at:
            return False
        if current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _resolve_workspace_root(workspace_dir: str | Path | None = None) -> Path:
    target = workspace_dir if workspace_dir is not None else _default_workspace_dir
    if target is None:
        raw_workspace = Path.cwd()
    else:
        if not isinstance(target, (str, Path)):
            raise TypeError("Workspace directory must be a string or Path.")
        if isinstance(target, str) and not target.strip():
            raise ValueError("Workspace directory path cannot be empty.")
        raw_workspace = Path(target)

    if raw_workspace.is_symlink():
        raise PermissionError(f"Workspace directory cannot be a symlink: {raw_workspace}")
    if not raw_workspace.exists():
        raise FileNotFoundError(str(raw_workspace))
    if not raw_workspace.is_dir():
        raise NotADirectoryError(str(raw_workspace))
    return raw_workspace.resolve()


def _resolve_workspace_path(path: str | Path, workspace_dir: str | Path | None = None) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("Path must be a string or Path.")
    if not str(path).strip():
        raise ValueError("Path cannot be empty.")

    workspace_root = _resolve_workspace_root(workspace_dir)
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else (workspace_root / raw_path)

    if _has_symlink_component(candidate, stop_at=workspace_root):
        raise PermissionError(f"Symlinks are not allowed: {path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise PermissionError(f"Path '{path}' is outside the workspace directory '{workspace_root}'.")
    if _has_symlink_component(resolved, stop_at=workspace_root):
        raise PermissionError(f"Symlinks are not allowed: {path}")
    return resolved


_PYTHON_BOOTSTRAP = (
    "import os, sys; sys.path.insert(0, os.getcwd()); "
    "exec(compile(sys.argv[1], '<run_python>', 'exec'), {'__name__': '__main__'})"
)

LANGUAGE_RUNNERS: dict[str, Callable[[str], list[str]]] = {
    "python": lambda code: [sys.executable, "-I", "-c", _PYTHON_BOOTSTRAP, code],
    "py": lambda code: [sys.executable, "-I", "-c", _PYTHON_BOOTSTRAP, code],
    "javascript": lambda code: ["node", "-e", code],
    "js": lambda code: ["node", "-e", code],
    "node": lambda code: ["node", "-e", code],
    "ruby": lambda code: ["ruby", "-e", code],
    "rb": lambda code: ["ruby", "-e", code],
    "php": lambda code: ["php", "-r", code],
    "lua": lambda code: ["lua", "-e", code],
    "perl": lambda code: ["perl", "-e", code],
    "r": lambda code: ["Rscript", "-e", code],
}

COMPILED_LANGUAGES: dict[str, tuple[str, str]] = {
    "cpp": ("g++", ".cpp"),
    "c++": ("g++", ".cpp"),
    "c": ("gcc", ".c"),
}


_validate_syntax = validate_syntax


@_handle_errors
def edit_file(
    path: str,
    target: str,
    replacement: str,
    workspace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Tworzy nowy plik kodu (gdy ``target=''``) lub edytuje unikalny fragment istniejącego pliku w ``workspace_dir``."""
    if not isinstance(target, str) or not isinstance(replacement, str):
        return {"success": False, "error": "Target and replacement must be strings."}

    file_path = _resolve_workspace_path(path, workspace_dir)

    if not file_path.exists():
        if not target:
            validate_syntax(file_path, replacement)
            write_res = write_file(str(file_path), replacement)
            if not write_res.get("success"):
                return write_res
            return {
                "success": True,
                "message": "File created and validated successfully.",
                "path": str(file_path),
                "created": True,
                "replacements": 1,
            }
        raise FileNotFoundError(str(file_path))
    if file_path.is_dir():
        raise IsADirectoryError(str(file_path))
    if not target:
        return {"success": False, "error": "Target fragment cannot be empty when editing an existing file."}

    raw_content = file_path.read_bytes().decode("utf-8")
    uses_crlf = "\r\n" in raw_content
    norm_content = raw_content.replace("\r\n", "\n")
    norm_target = target.replace("\r\n", "\n")
    norm_replacement = replacement.replace("\r\n", "\n")

    occurrences = norm_content.count(norm_target)
    if occurrences == 0:
        return {"success": False, "error": "Target fragment not found in file."}
    if occurrences > 1:
        return {
            "success": False,
            "error": f"Target fragment occurs {occurrences} times (ambiguous match). Provide more surrounding context.",
        }

    updated_norm = norm_content.replace(norm_target, norm_replacement, 1)
    _validate_syntax(file_path, updated_norm)

    final_content = updated_norm.replace("\n", "\r\n") if uses_crlf else updated_norm
    file_path.write_bytes(final_content.encode("utf-8"))
    return {"success": True, "message": "File edited successfully.", "path": str(file_path), "replacements": 1}


apply_patch = edit_file


def _build_run_result(
    stdout: str | bytes | None = "",
    stderr: str | bytes | None = "",
    *,
    exit_code: int | None = None,
    error: str | None = None,
    limit: int = MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    def clip(raw: str | bytes | None) -> tuple[str, bool]:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
        return (text[:limit] + "\n...[Output truncated]", True) if len(text) > limit else (text, False)

    out, out_trunc = clip(stdout)
    err, err_trunc = clip(stderr)
    result: dict[str, Any] = {
        "success": error is None and exit_code == 0,
        "stdout": out,
        "stderr": err,
        "exit_code": exit_code,
        "truncated": out_trunc or err_trunc,
        "isolation_warning": ISOLATION_WARNING,
    }
    if error is not None:
        result["error"] = error
    return result


def _execute_subprocess(
    cmd: list[str], cwd: Path, timeout_val: float, max_output_chars: int
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_val,
        )
    except subprocess.TimeoutExpired as error:
        return _build_run_result(
            error.stdout, error.stderr,
            error=f"Execution timed out after {timeout_val} seconds.",
            limit=max_output_chars,
        )
    except OSError as error:
        return _build_run_result(error=f"Failed to start {cmd[0]} process: {error}")

    result = _build_run_result(
        completed.stdout, completed.stderr, exit_code=completed.returncode, limit=max_output_chars
    )
    if completed.returncode != 0:
        err_detail = result["stderr"].strip()
        result["error"] = (
            f"Process exited with code {completed.returncode}: {err_detail}"
            if err_detail
            else f"Process exited with code {completed.returncode}"
        )
    return result


def run_python(
    code: str,
    timeout: float = DEFAULT_TIMEOUT,
    workspace_dir: str | Path | None = None,
    *,
    language: str = "python",
    allow_execution: bool | None = None,
    approval_callback: Callable[[str], bool] | None = None,
    max_output_chars: int = MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Uruchamia kod w wybranym języku (domyślnie Python) w podprocesie po uzyskaniu zgody użytkownika (nie jest to pełny sandbox)."""
    Logger.printWarning(ISOLATION_WARNING)

    if not isinstance(code, str) or not code.strip():
        return _build_run_result(error="Code must be a non-empty string.")
    if not isinstance(language, str) or not language.strip():
        return _build_run_result(error="Language must be a non-empty string.")
    lang_key = language.strip().lower()
    runner = LANGUAGE_RUNNERS.get(lang_key)
    compiled_spec = COMPILED_LANGUAGES.get(lang_key)
    if runner is None and compiled_spec is None:
        supported = ", ".join(sorted({*LANGUAGE_RUNNERS, *COMPILED_LANGUAGES}))
        return _build_run_result(error=f"Unsupported language '{language}'. Allowed languages: {supported}.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return _build_run_result(error="Timeout must be a number.")
    timeout_val = float(timeout)
    if timeout_val <= 0 or timeout_val > MAX_TIMEOUT:
        return _build_run_result(error=f"Timeout must be between 0 and {MAX_TIMEOUT} seconds.")
    if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int) or max_output_chars <= 0:
        return _build_run_result(error="max_output_chars must be a positive integer.")

    try:
        workspace_root = _resolve_workspace_root(workspace_dir)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError, ValueError, TypeError) as error:
        return _build_run_result(error=str(error))

    cb = approval_callback if approval_callback is not None else _default_approval_callback
    allowed = _default_allow_execution if allow_execution is None else allow_execution
    if cb is not None:
        try:
            approved = bool(cb(code))
        except Exception as error:
            return _build_run_result(error=f"Approval check failed: {error}")
        if not approved:
            return _build_run_result(error="Permission denied: User rejected code execution.")
    elif not allowed:
        return _build_run_result(error="Permission denied: Explicit user approval is required before running code.")

    if compiled_spec is not None:
        compiler, suffix = compiled_spec
        if shutil.which(compiler) is None:
            return _build_run_result(
                error=f"Compiler '{compiler}' for language '{language}' was not found in PATH."
            )
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / f"snippet{suffix}"
            bin_path = tmp_path / ("snippet.exe" if sys.platform == "win32" else "snippet")
            src_path.write_text(code, encoding="utf-8")
            compile_res = _execute_subprocess(
                [compiler, str(src_path), "-o", str(bin_path)],
                workspace_root,
                timeout_val,
                max_output_chars,
            )
            if not compile_res["success"]:
                compile_res["error"] = f"Compilation failed: {compile_res.get('error', '')}".strip()
                return compile_res
            return _execute_subprocess([str(bin_path)], workspace_root, timeout_val, max_output_chars)

    assert runner is not None
    cmd = runner(code)
    if lang_key not in ("python", "py") and shutil.which(cmd[0]) is None:
        return _build_run_result(
            error=f"Interpreter '{cmd[0]}' for language '{language}' was not found in PATH."
        )
    return _execute_subprocess(cmd, workspace_root, timeout_val, max_output_chars)


run_code = run_python
AVAILABLE_CODE_TOOLS = {"edit_file": edit_file, "run_python": run_python}


def register_code_tools(
    tool_handle: ToolHandle,
    *,
    workspace_dir: str | Path | None = None,
    allow_execution: bool = False,
    approval_callback: Callable[[str], bool] | None = None,
) -> None:
    """Rejestruje narzędzia kodowania w ``ToolHandle`` z polityką kontrolowaną po stronie hosta."""

    def _edit_file_tool(path: str, target: str, replacement: str) -> dict[str, Any]:
        return edit_file(path, target, replacement, workspace_dir=workspace_dir)

    def _run_python_tool(
        code: str, timeout: float = DEFAULT_TIMEOUT, language: str = "python"
    ) -> dict[str, Any]:
        return run_python(
            code, timeout, workspace_dir,
            language=language, allow_execution=allow_execution, approval_callback=approval_callback,
        )

    tool_handle.create_tool(
        name="edit_file",
        description=(
            "Primary tool for creating new code files (pass empty target='' and full code in replacement) "
            "or editing a unique target fragment inside an existing UTF-8 file within the workspace directory. "
            "Automatically validates syntax for .py, .json, .c, and .cpp files before saving. "
            "Always prefer edit_file over write_file for source code files."
        ),
        properties={
            "path": {"type": "string", "description": "Path to the file within the workspace directory."},
            "target": {
                "type": "string",
                "description": "Exact existing text fragment to replace (must occur uniquely), or '' (empty string) to create a new file.",
            },
            "replacement": {"type": "string", "description": "New text fragment, or full file content when creating a new file."},
        },
        required=["path", "target", "replacement"],
        function=_edit_file_tool,
    )
    tool_handle.create_tool(
        name="run_python",
        description=(
            "Runs code in a supported language (default 'python'; also supports cpp/c++, c, "
            "javascript/js/node, ruby, php, lua, perl, r) as a separate subprocess with timeout "
            "and output limits in the workspace directory. Requires host-side user approval; not a full OS sandbox."
        ),
        properties={
            "code": {"type": "string", "description": "Source code to execute."},
            "language": {
                "type": "string",
                "description": "Programming language (default 'python'; e.g. 'python', 'cpp', 'c', 'javascript', 'ruby', 'php', 'lua').",
            },
            "timeout": {"type": "number", "description": f"Timeout in seconds (0 < timeout <= {MAX_TIMEOUT}, default {DEFAULT_TIMEOUT})."},
        },
        required=["code"],
        function=_run_python_tool,
    )
