from functools import wraps
from pathlib import Path
from handlers.ToolHandle import ToolHandle
from datetime import date, datetime, time, timezone
from fnmatch import fnmatchcase
from stat import S_ISDIR, S_ISREG
import shutil

# Wrapper zwracający słownik z kluczem "success" i ewentualnym komunikatem o błędzie.
def _handle_errors(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
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
        except (OSError, ValueError, shutil.Error) as error:
            message = f"Operation failed: {error}"

        return {"success": False, "error": message}

    return wrapper


@_handle_errors
def create_file(path: str, content: str = ""):
    file_path = Path(path)

    # Tryb "x" tworzy nowy plik i nie nadpisuje istniejącego.
    with file_path.open("x", encoding="utf-8") as file:
        file.write(content)

    return {
        "success": True,
        "message": "File created successfully.",
        "path": str(file_path),
    }


@_handle_errors
def read_file(path: str):
    file_path = Path(path)

    return {
        "success": True,
        "path": str(file_path),
        "content": file_path.read_text(encoding="utf-8"),
    }


@_handle_errors
def write_file(path: str, content: str):
    file_path = Path(path)

    # Zastępuje zawartość albo tworzy plik, jeśli go nie ma.
    file_path.write_text(content, encoding="utf-8")

    return {
        "success": True,
        "message": "File content written successfully.",
        "path": str(file_path),
    }


@_handle_errors
def move_file(source: str, destination: str):
    source_path = Path(source)
    destination_path = Path(destination)

    if not source_path.exists():
        raise FileNotFoundError(str(source_path))

    if not source_path.is_file():
        return {
            "success": False,
            "error": f"Source is not a file: {source_path}",
        }

    # destination oznacza pełną ścieżkę docelowego pliku.
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(str(destination_path))

    # shutil.move obsługuje także przenoszenie między dyskami.
    shutil.move(str(source_path), str(destination_path))

    return {
        "success": True,
        "message": "File moved successfully.",
        "source": str(source_path),
        "destination": str(destination_path),
    }


@_handle_errors
def delete_file(path: str):
    file_path = Path(path)
    file_path.unlink()

    return {
        "success": True,
        "message": "File deleted successfully.",
        "path": str(file_path),
    }


@_handle_errors
def list_directory(path: str):
    directory_path = Path(path)

    entries = [
        {
            "name": entry.name,
            "path": str(entry),
            "type": "directory" if entry.is_dir() else "file",
        }
        for entry in sorted(
            directory_path.iterdir(),
            key=lambda entry: entry.name.casefold(),
        )
    ]

    return {
        "success": True,
        "path": str(directory_path),
        "entries": entries,
    }

@_handle_errors
def search_in_files(path: str, text: str, case_sensitive: bool = False):
    search_path = Path(path)

    if not text:
        return {
            "success": False,
            "error": "Text to search for cannot be empty.",
        }

    if not search_path.exists():
        raise FileNotFoundError(str(search_path))

    if search_path.is_file():
        files = [search_path]
    elif search_path.is_dir():
        files = (
            entry
            for entry in search_path.rglob("*")
            if entry.is_file()
        )
    else:
        return {
            "success": False,
            "error": "Provided path is not a file or directory.",
        }

    searched_text = text if case_sensitive else text.casefold()
    matches = []
    skipped_files = []

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            skipped_files.append({
                "path": str(file_path),
                "error": f"Cannot read file as UTF-8 text: {error}",
            })
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            searched_line = line if case_sensitive else line.casefold()

            if searched_text in searched_line:
                matches.append({
                    "path": str(file_path),
                    "line_number": line_number,
                    "line": line,
                })

    return {
        "success": True,
        "matches_count": len(matches),
        "matches": matches,
        "skipped_files": skipped_files,
    }

def _get_path_info(path: Path):
    # Wspólna funkcja pomocnicza do odczytu metadanych.
    file_stat = path.stat()

    is_file = S_ISREG(file_stat.st_mode)
    is_directory = S_ISDIR(file_stat.st_mode)

    return {
        "path": str(path.absolute()),
        "name": path.name,
        "extension": path.suffix.lower() if is_file else "",
        "size_bytes": file_stat.st_size if is_file else None,
        "modified_at": datetime.fromtimestamp(
            file_stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "is_file": is_file,
        "is_directory": is_directory,
    }

@_handle_errors
def get_file_info(path: str):
    return {
        "success": True,
        **_get_path_info(Path(path)),
    }

@_handle_errors
def find_files(
    path: str,
    name_pattern: str = "*",
    extensions= None,
    modified_before: str | None = None,
    recursive: bool = True,
):
    directory_path = Path(path)

    # stat() zgłosi błąd, jeśli ścieżka nie istnieje
    # albo nie można odczytać jej metadanych.
    directory_stat = directory_path.stat()

    if not S_ISDIR(directory_stat.st_mode):
        return {
            "success": False,
            "error": f"Search path is not a directory: {directory_path}",
        }

    cutoff = None

    if modified_before is not None:
        try:
            cutoff_date = date.fromisoformat(modified_before)
        except ValueError:
            return {
                "success": False,
                "error": "modified_before must be a valid date in YYYY-MM-DD format.",
            }

        # Szukamy plików zmodyfikowanych przed początkiem tego dnia w UTC.
        cutoff = datetime.combine(
            cutoff_date,
            time.min,
            tzinfo=timezone.utc,
        )

    normalized_extensions = None

    if extensions is not None:
        normalized_extensions = {
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in extensions
        }

    entries = (
        directory_path.rglob("*")
        if recursive
        else directory_path.iterdir()
    )

    files = []
    skipped_files = []

    for entry in entries:
        # Dopasowanie nazwy bez rozróżniania wielkości liter.
        if not fnmatchcase(entry.name.casefold(), name_pattern.casefold()):
            continue

        if (
            normalized_extensions is not None
            and entry.suffix.lower() not in normalized_extensions
        ):
            continue

        try:
            info = _get_path_info(entry)
        except OSError as error:
            skipped_files.append({
                "path": str(entry.absolute()),
                "error": f"Cannot read path metadata: {error}",
            })
            continue

        if not info["is_file"]:
            continue

        modified_at = datetime.fromisoformat(info["modified_at"])

        if cutoff is not None and modified_at >= cutoff:
            continue

        files.append(info)

    files.sort(key=lambda info: info["path"])

    return {
        "success": True,
        "files_count": len(files),
        "files": files,
        "skipped_files": skipped_files,
    }

# Rejestracja narzędzi w ToolHandle
def register_tools(tool_handle):
    path_property = {
        "type": "string",
        "description": "Path to the file or directory.",
    }

    content_property = {
        "type": "string",
        "description": "Text content to write using UTF-8 encoding.",
    }

    definitions = [
        {
            "name": "create_file",
            "description": (
                "Creates a new UTF-8 text file. "
                "Fails if the path already exists. "
                "The parent directory must exist."
            ),
            "properties": {
                "path": path_property,
                "content": content_property,
            },
            "required": ["path"],
        },
        {
            "name": "read_file",
            "description": "Reads the content of a UTF-8 text file.",
            "properties": {"path": path_property},
            "required": ["path"],
        },
        {
            "name": "write_file",
            "description": (
                "Replaces a file's entire content with UTF-8 text. "
                "Creates the file if it does not exist. "
                "The parent directory must exist."
            ),
            "properties": {
                "path": path_property,
                "content": content_property,
            },
            "required": ["path", "content"],
        },
        {
            "name": "move_file",
            "description": (
                "Moves or renames a file. "
                "The destination must be a full file path and must not exist. "
                "The destination parent directory must exist."
            ),
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to the existing source file.",
                },
                "destination": {
                    "type": "string",
                    "description": "Full destination path including the file name.",
                },
            },
            "required": ["source", "destination"],
        },
        {
            "name": "delete_file",
            "description": "Deletes a file. Does not delete directories.",
            "properties": {"path": path_property},
            "required": ["path"],
        },
        {
            "name": "list_directory",
            "description": (
                "Lists the immediate contents of a directory without recursion."
            ),
            "properties": {"path": path_property},
            "required": ["path"],
        },
        {
            "name": "search_in_files",
            "description": (
                "Searches for a text string in files within a directory or a single file. "
                "Returns the file path, line number, and line content for each match. "
                "Skips files that cannot be read as UTF-8 text."
            ),
            "properties": {
                "path": path_property,
                "text": {
                    "type": "string",
                    "description": "Text to search for.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": (
                        "If true, the search is case-sensitive. "
                        "Defaults to false (case-insensitive)."
                    ),
                },
            },
            "required": ["path", "text"],
        },
        {
            "name": "get_file_info",
            "description": (
                "Returns metadata for a file or directory: absolute path, name, "
                "extension, file size in bytes, last modification time in UTC, "
                "and whether the path is a file or directory. "
                "Directory size is returned as null."
            ),
            "properties": {
                "path": path_property,
            },
            "required": ["path"],
        },
        {
            "name": "find_files",
            "description": (
                "Finds files in a directory using optional filename, extension, "
                "and last modification date filters. All filters must match. "
                "Returns absolute file paths and metadata. "
                "Does not search file contents. "
                "Searches subdirectories by default."
            ),
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to search.",
                },
                "name_pattern": {
                    "type": "string",
                    "description": (
                        "Case-insensitive filename pattern, such as report*.pdf. "
                        "Supports * for any sequence of characters and ? for one "
                        "character. Defaults to *."
                    ),
                },
                "extensions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Allowed extensions, such as ['.pdf', '.txt']. "
                        "Matching is case-insensitive. The leading dot is optional. "
                        "If omitted, all extensions are allowed."
                    ),
                },
                "modified_before": {
                    "type": "string",
                    "description": (
                        "Select files last modified strictly before midnight UTC "
                        "at the start of this date. Format: YYYY-MM-DD."
                    ),
                },
                "recursive": {
                    "type": "boolean",
                    "description": (
                        "Whether to search subdirectories. Defaults to true."
                    ),
                },
            },
            "required": ["path"],
        },       
    ]

    for definition in definitions:
        tool_handle.create_tool(**definition)


tools = ToolHandle()
register_tools(tools)
