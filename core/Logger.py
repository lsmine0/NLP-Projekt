import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from colorama import Fore, Style, init

init(autoreset=True)
VERBOSE = 3

_FILE_LOGGER = logging.getLogger("jarvis.file")
_FILE_LOGGER.setLevel(logging.INFO)
_FILE_LOGGER.propagate = False
_FILE_PATH: Path | None = None


def configure_file_logging(path: str | Path | None = None) -> Path | None:
    """Enable persistent logging, e.g. data/logs/app.log.

    ``NLP_LOG_FILE`` overrides the supplied path. Passing ``None`` without the
    environment variable disables file logging and removes the previous handler.
    Console output still respects ``VERBOSE``.
    """
    global _FILE_PATH

    env_path = os.environ.get("NLP_LOG_FILE")
    if env_path:
        path = env_path
    if path is None:
        for handler in list(_FILE_LOGGER.handlers):
            _FILE_LOGGER.removeHandler(handler)
            handler.close()
        _FILE_PATH = None
        return None

    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(mode=0o600, exist_ok=True)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        handler = RotatingFileHandler(
            target,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        for old in list(_FILE_LOGGER.handlers):
            _FILE_LOGGER.removeHandler(old)
            old.close()
        _FILE_LOGGER.addHandler(handler)
        _FILE_LOGGER.setLevel(logging.INFO)
        _FILE_PATH = target
        return target
    except OSError:
        _FILE_PATH = None
        return None


def log_file_path() -> Path | None:
    return _FILE_PATH


def _file_log(level: int, message: str) -> None:
    if _FILE_PATH is None:
        return
    try:
        _FILE_LOGGER.log(level, message)
    except Exception:
        # Logging must never break the agent loop.
        pass


def printInfo(message: str) -> None:
    _file_log(logging.INFO, message)
    if VERBOSE >= 3:
        print(Fore.BLUE + "[Info]" + Style.RESET_ALL + f" - {message}")


def printThinking(message: str) -> None:
    # Reasoning can contain sensitive context; do not persist it to disk.
    if VERBOSE >= 3:
        print(Fore.MAGENTA + "[Thinking]" + Style.RESET_ALL + f" - {message}")


def printThinkingStream(message: str, start: bool = False) -> None:
    if VERBOSE >= 3:
        if start:
            print(Fore.MAGENTA + "[Thinking] " + Style.RESET_ALL, end="", flush=True)
        print(message, end="", flush=True)


def printThinkingEnd() -> None:
    if VERBOSE >= 3:
        print()


def printWarning(message: str) -> None:
    _file_log(logging.WARNING, message)
    if VERBOSE >= 2:
        print(Fore.YELLOW + "[Warning]" + Style.RESET_ALL + f" - {message}")


def printError(message: str) -> None:
    _file_log(logging.ERROR, message)
    if VERBOSE >= 1:
        print(Fore.RED + "[Error]" + Style.RESET_ALL + f" - {message}")


if __name__ == "__main__":
    printInfo("info Test")
    printThinking("thinking Test")
    printWarning("Warning Test")
    printError("Error Test")
