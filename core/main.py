from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import Logger
from core.AgentAdapter import (
    AgentBase,
    OllamaAgent,
    STTBase,
    StubSTT,
    StubTTS,
    TTSBase,
    create_agent,
    create_stt,
    create_tts,
)
from core.ChatStore import ChatStore, SettingsStore
from handlers.OllamaHandle import OllamaHandler
from handlers.ToolHandle import ToolHandle
from tools.FileTools import register_tools as register_file_tools
from tools.MathTools import register_math_tools
from tools.WebSearchTools import register_web_search_tools


class StartupError(RuntimeError):
    """Czytelny błąd startu aplikacji."""


class DisabledSTT(STTBase):
    """Blokuje nagrywanie, gdy uruchomiono aplikację z flagą --no-audio."""

    def start(self) -> None:
        raise RuntimeError("Obsługa audio jest wyłączona (--no-audio).")

    def stop(self) -> str:
        raise RuntimeError("Obsługa audio jest wyłączona (--no-audio).")


class DisabledTTS(TTSBase):
    """Blokuje syntezę mowy, gdy uruchomiono aplikację z flagą --no-audio."""

    def speak(self, text: str) -> None:
        raise RuntimeError("Obsługa audio jest wyłączona (--no-audio).")

    def stop(self) -> None:
        pass


@dataclass
class AppRuntime:
    """Kontener zależności aplikacji budowany w main.py."""

    settings_store: SettingsStore
    chat_store: ChatStore
    tool_handle: ToolHandle
    provider: OllamaHandler
    stt: STTBase
    tts: TTSBase
    agent_factory: Callable[[dict], AgentBase]
    audio_enabled: bool = True
    mcp_clients: list[Any] = field(default_factory=list)

    @property
    def settings(self) -> dict:
        return self.settings_store.get()

    def close(self) -> None:
        """Zamyka zasoby aplikacji: TTS, STT, MCP oraz sesje."""
        for audio_component in (self.tts, self.stt):
            for method_name in ("close", "stop"):
                method = getattr(audio_component, method_name, None)
                if callable(method):
                    try:
                        method()
                        break
                    except Exception:
                        pass

        for mcp_client in self.mcp_clients:
            close_fn = getattr(mcp_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        self.mcp_clients.clear()

        if "core.WebGui" in sys.modules:
            try:
                sys.modules["core.WebGui"].sessions.clear()
            except Exception:
                pass


CONFIG_DESCRIPTIONS: dict[str, str] = {
    "ollama_endpoint": "Adres serwera modeli (np. http://localhost:11434/v1)",
    "ollama_model": "Nazwa modelu używanego przez agenta (np. qwen3.5:4b)",
    "max_chats": "Maksymalna liczba zapisanych rozmów (liczba całkowita, np. 50)",
    "max_age_days": "Maksymalny wiek zapisanych rozmów w dniach (np. 30)",
    "tts_enabled": "Automatyczne czytanie odpowiedzi na głos (true / false)",
    "stt_auto_send": "Automatyczne wysyłanie tekstu po nagraniu głosu (true / false)",
    "theme": "Motyw kolorystyczny GUI (dark / light)",
    "show_history": "Pokazywanie paska historii rozmów w GUI (true / false)",
}


def _parse_config_value(key: str, raw_val: str) -> Any:
    default_val = SettingsStore.defaults().get(key)
    if isinstance(default_val, bool):
        low = raw_val.strip().lower()
        if low in ("true", "1", "yes", "tak", "on"):
            return True
        if low in ("false", "0", "no", "nie", "off"):
            return False
        raise StartupError(f"Wartość dla '{key}' musi być typu logicznego (true/false), otrzymano: '{raw_val}'.")
    if isinstance(default_val, int):
        try:
            return int(raw_val.strip())
        except ValueError as exc:
            raise StartupError(f"Wartość dla '{key}' musi być liczbą całkowitą, otrzymano: '{raw_val}'.") from exc
    return raw_val.strip()


def print_config_help(settings_store: SettingsStore) -> None:
    current = settings_store.get()
    print(f"\n=== Aktualna konfiguracja ({settings_store.path}) ===")
    for key, val in current.items():
        desc = CONFIG_DESCRIPTIONS.get(key, "")
        print(f"  {key:<18} = {val!r:<32} # {desc}")

    print("\nJak używać flagi --config:")
    print("  1. Podgląd aktualnych ustawień:")
    print("       python main.py --config")
    print("  2. Zmiana pojedynczego ustawienia (klucz=wartość):")
    print("       python main.py --config ollama_model=qwen3.5:4b")
    print("       python main.py --config ollama_endpoint=http://localhost:11434/v1")
    print("       python main.py --config tts_enabled=true")
    print("  3. Uruchomienie z własnym plikiem JSON:")
    print("       python main.py --config sciezka/do/settings.json\n")


# 1. Config
def build_config(config_arg: str | None = None) -> tuple[SettingsStore, ChatStore]:
    path: Path | None = None
    inline_updates: dict[str, Any] = {}

    if config_arg and config_arg != "__show__":
        if "=" in config_arg and not Path(config_arg).exists():
            key, raw_val = config_arg.split("=", 1)
            key = key.strip()
            if key not in SettingsStore.defaults():
                valid_keys = ", ".join(SettingsStore.defaults().keys())
                raise StartupError(
                    f"Nieznany klucz konfiguracji '{key}'. Dostępne klucze: {valid_keys}"
                )
            inline_updates[key] = _parse_config_value(key, raw_val)
        else:
            path = Path(config_arg).expanduser().resolve()
            if path.is_dir():
                raise StartupError(f"Podana ścieżka --config jest katalogiem, a nie plikiem: {path}")

    settings_store = SettingsStore(path=path)
    if inline_updates:
        settings_store.update(inline_updates)
        for k, v in inline_updates.items():
            Logger.printInfo(f"Zaktualizowano ustawienie: {k} = {v!r} (zapisano w {settings_store.path})")

    chat_dir = path.parent / "chats" if path else None
    chat_store = ChatStore(directory=chat_dir)
    return settings_store, chat_store


# 2. ToolHandle
def build_tools() -> ToolHandle:
    tool_handle = ToolHandle()
    register_file_tools(tool_handle)
    register_math_tools(tool_handle, overwrite=True)
    register_web_search_tools(tool_handle, overwrite=True)
    return tool_handle


# 3. Provider / Model
def build_provider(
    settings: dict,
    tool_handle: ToolHandle,
    provider_name: str = "ollama",
    require_connection: bool = False,
) -> OllamaHandler:
    provider_norm = (provider_name or "ollama").strip().lower()
    if provider_norm not in ("ollama", "openai", "local"):
        raise StartupError(f"Nieobsługiwany provider: '{provider_name}'.")

    endpoint = str(settings.get("ollama_endpoint") or "").strip()
    if not endpoint:
        raise StartupError("Brak skonfigurowanego adresu serwera modeli (ollama_endpoint).")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OLLAMA_API_KEY") or "ollam"
    if provider_norm == "openai" and api_key == "ollam":
        raise StartupError("Brak klucza API dla providera 'openai' (ustaw OPENAI_API_KEY).")

    handler = OllamaHandler(base_url=endpoint, key=api_key)
    handler.ToolsHandle = tool_handle

    try:
        models = [m.id for m in handler.GetModels()]
    except Exception as exc:
        msg = f"Brak połączenia z serwerem modeli ({endpoint}): {exc}"
        if require_connection:
            raise StartupError(msg) from exc
        Logger.printWarning(msg)
        return handler

    if not models:
        msg = f"Brak dostępnych modeli na serwerze ({endpoint})."
        if require_connection:
            raise StartupError(msg)
        Logger.printWarning(msg)
        return handler

    target_model = settings.get("ollama_model") or models[0]
    if not handler.SetModel(target_model):
        msg = f"Model '{target_model}' nie jest dostępny na serwerze ({endpoint})."
        if require_connection:
            raise StartupError(msg)
        Logger.printWarning(msg)

    return handler


# 4. Audio (STT / TTS)
def build_audio(no_audio: bool = False) -> tuple[STTBase, TTSBase]:
    if no_audio:
        Logger.printInfo("Flaga --no-audio: funkcje głosowe (STT/TTS) zostały zablokowane.")
        return DisabledSTT(), DisabledTTS()

    try:
        stt = create_stt()
    except Exception as exc:
        Logger.printWarning(f"Brak zależności audio dla STT ({exc}) — używam StubSTT.")
        stt = StubSTT()

    try:
        tts = create_tts()
    except Exception as exc:
        Logger.printWarning(f"Brak zależności audio dla TTS ({exc}) — używam StubTTS.")
        tts = StubTTS()

    return stt, tts


# 5. Spięcie zależności w jednym miejscu: config -> ToolHandle -> provider -> agent -> GUI/CLI
def build_runtime(args: argparse.Namespace) -> AppRuntime:
    settings_store, chat_store = build_config(args.config)
    settings = settings_store.get()

    tool_handle = build_tools()
    provider = build_provider(
        settings,
        tool_handle=tool_handle,
        provider_name=args.provider,
        require_connection=bool(args.cli or args.benchmark),
    )
    stt, tts = build_audio(no_audio=args.no_audio)

    # Przekazanie wspólnego ToolHandle do ujednoliconego agenta
    OllamaAgent.default_tool_handle = tool_handle

    def agent_factory(current_settings: dict | None = None) -> AgentBase:
        cfg = current_settings or settings_store.get()
        return create_agent(cfg)

    return AppRuntime(
        settings_store=settings_store,
        chat_store=chat_store,
        tool_handle=tool_handle,
        provider=provider,
        stt=stt,
        tts=tts,
        agent_factory=agent_factory,
        audio_enabled=not args.no_audio,
    )


def run_cli(runtime: AppRuntime) -> int:
    """Tryb CLI korzystający z tej samej pętli agenta co GUI."""
    Logger.VERBOSE = 1
    agent = runtime.agent_factory(runtime.settings)
    model_name = runtime.provider.model or runtime.settings.get("ollama_model") or "domyślny"
    audio_label = "AKTYWNE" if runtime.audio_enabled else "ZABLOKOWANE (--no-audio)"

    show_thinking_live = False
    last_thinking = ""

    print(f"=== Jarvis CLI (model: {model_name} | audio: {audio_label}) ===")
    print("Komendy: /voice (nagraj głos), /think (pokaż ostatnie myślenie), /think live,")
    print("         /logs (przełącz logi systemowe), /reset (nowa rozmowa), exit / quit\n")

    while True:
        try:
            user_input = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "/exit", "/quit"):
            break
        if cmd == "/reset":
            agent.reset()
            last_thinking = ""
            print("[System] Historia rozmowy została wyczyszczona.\n")
            continue
        if cmd == "/voice":
            if not runtime.audio_enabled:
                print("[System] Funkcje audio (STT/TTS) są zablokowane flagą --no-audio.\n")
                continue
            try:
                runtime.stt.start()
                input("[STT] Nagrywanie... Naciśnij Enter, aby zakończyć.")
                user_input = runtime.stt.stop().strip()
                print(f"[STT] Rozpoznano: {user_input}")
                if not user_input:
                    continue
            except Exception as exc:
                Logger.printError(f"STT: {exc}")
                continue
        if cmd == "/think":
            if last_thinking:
                print(f"\n--- Myślenie modelu (ostatnia odpowiedź) ---\n{last_thinking}\n--------------------------------------------\n")
            else:
                print("[System] Brak zapisanego procesu myślenia z ostatniej odpowiedzi.\n")
            continue
        if cmd == "/think live":
            show_thinking_live = not show_thinking_live
            state = "WŁĄCZONY" if show_thinking_live else "WYŁĄCZONY"
            print(f"[System] Podgląd myślenia na żywo: {state}.\n")
            continue
        if cmd == "/logs":
            Logger.VERBOSE = 3 if Logger.VERBOSE < 3 else 1
            state = "WŁĄCZONE" if Logger.VERBOSE == 3 else "WYŁĄCZONE"
            print(f"[System] Logi diagnostyczne: {state}.\n")
            continue

        thinking_parts: list[str] = []
        reply_parts: list[str] = []
        in_thinking = False
        header_printed = False

        for event in agent.stream(user_input):
            if event["type"] == "thinking":
                text = event["text"]
                thinking_parts.append(text)
                if show_thinking_live:
                    if not in_thinking:
                        print("\n[Thinking] ", end="", flush=True)
                        in_thinking = True
                    print(text, end="", flush=True)
                elif not in_thinking:
                    print("\rAsystent: (myślenie...)", end="", flush=True)
                    in_thinking = True

            elif event["type"] == "token":
                if in_thinking:
                    if show_thinking_live:
                        print("\n")
                    else:
                        print("\r" + " " * 30 + "\r", end="", flush=True)
                    in_thinking = False
                if not header_printed:
                    print("Asystent: ", end="", flush=True)
                    header_printed = True
                token_text = event["text"]
                reply_parts.append(token_text)
                print(token_text, end="", flush=True)

            elif event["type"] == "error":
                if in_thinking:
                    print("\r" + " " * 30 + "\r", end="", flush=True)
                    in_thinking = False
                print()
                Logger.printError(event["message"])

        if in_thinking and not header_printed:
            print("\r" + " " * 30 + "\rAsystent: (brak treści odpowiedzi)", end="", flush=True)

        last_thinking = "".join(thinking_parts).strip()
        if last_thinking and not show_thinking_live:
            print("\n  [Ukryto proces myślenia — wpisz /think, aby wyświetlić]\n")
        else:
            print("\n")

        full_reply = "".join(reply_parts).strip()
        if full_reply and runtime.audio_enabled and runtime.settings.get("tts_enabled"):
            try:
                runtime.tts.speak(full_reply)
            except Exception as exc:
                Logger.printError(f"TTS: {exc}")

    return 0


def run_benchmark(runtime: AppRuntime, extra_args: list[str]) -> int:
    """Uruchamia BenchmarkRunner z konfiguracją z main.py."""
    import BenchmarkRunner

    model_name = runtime.provider.model or runtime.settings.get("ollama_model")
    if not model_name:
        raise StartupError("Tryb --benchmark wymaga skonfigurowanego modelu w ustawieniach.")

    old_argv = sys.argv
    try:
        sys.argv = [
            "BenchmarkRunner.py",
            "--model",
            model_name,
            "--endpoint",
            runtime.provider.base_url,
        ] + extra_args
        return int(BenchmarkRunner.main() or 0)
    finally:
        sys.argv = old_argv


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    epilog = (
        "Przykłady użycia:\n"
        "  python main.py                                    # uruchomienie GUI (domyślnie)\n"
        "  python main.py --cli                              # uruchomienie czatu w terminalu\n"
        "  python main.py --config                           # wyświetlenie aktualnych ustawień i pomocy\n"
        "  python main.py --config ollama_model=qwen3.5:4b   # zmiana modelu w ustawieniach i start\n"
        "  python main.py --config moj_config.json           # wczytanie ustawień ze wskazanego pliku JSON\n"
        "  python main.py --no-audio                         # uruchomienie bez mikrofonu (STT) i głosu (TTS)\n"
        "  python main.py --benchmark                        # uruchomienie testów BenchmarkRunner\n"
    )
    parser = argparse.ArgumentParser(
        description="Główny punkt wejścia NLP-Projekt.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="Uruchom GUI (domyślnie).")
    mode.add_argument("--cli", action="store_true", help="Uruchom tryb konsolowy CLI.")
    mode.add_argument("--benchmark", action="store_true", help="Uruchom benchmark.")

    parser.add_argument(
        "--config",
        nargs="?",
        const="__show__",
        default=None,
        metavar="PLIK_LUB_KLUCZ=WARTOŚĆ",
        help=(
            "Bez argumentu: wyświetla aktualne ustawienia i dostępne opcje. "
            "Z argumentem: ścieżka do pliku JSON (np. data/settings.json) "
            "lub zmiana ustawienia w formacie klucz=wartość (np. ollama_model=qwen3.5:4b)."
        ),
    )
    parser.add_argument("--no-audio", action="store_true", help="Wyłącz obsługę STT i TTS.")
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        metavar="NAZWA",
        help="Provider modelu: ollama (domyślnie), openai, local.",
    )

    args, extra = parser.parse_known_args(argv)
    if extra and not args.benchmark:
        parser.error(f"Nieznane argumenty: {' '.join(extra)}")

    if not args.cli and not args.benchmark:
        args.gui = True

    return args, extra


def main(argv: list[str] | None = None) -> int:
    args, extra_args = parse_args(argv)
    if args.config == "__show__":
        print_config_help(SettingsStore())
        return 0
    if args.cli:
        Logger.VERBOSE = 1
    runtime: AppRuntime | None = None

    try:
        runtime = build_runtime(args)

        if args.benchmark:
            return run_benchmark(runtime, extra_args)

        if args.cli:
            return run_cli(runtime)

        from core.WebGui import start_gui

        start_gui(
            chat_store=runtime.chat_store,
            config_store=runtime.settings_store,
            stt_handle=runtime.stt,
            tts_handle=runtime.tts,
            custom_agent_factory=runtime.agent_factory,
            audio_enabled=runtime.audio_enabled,
        )
        return 0

    except StartupError as err:
        Logger.printError(str(err))
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
