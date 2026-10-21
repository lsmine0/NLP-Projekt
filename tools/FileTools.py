from functools import wraps
from pathlib import Path
from ToolHandle import ToolHandle
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
    ]

    for definition in definitions:
        tool_handle.create_tool(**definition)


tools = ToolHandle()
register_tools(tools)
