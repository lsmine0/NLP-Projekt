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


class RealSTTAdapter(STTBase):
    """Adapter mapujący interfejs STTBase na istniejący STTHandle."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def start(self) -> None:
        self.handle.start_recording()

    def stop(self) -> str:
        return self.handle.stop_recording()


class RealTTSAdapter(TTSBase):
    """Adapter mapujący interfejs TTSBase na istniejący TTSHandle."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def speak(self, text: str) -> None:
        self.handle.speak(text)

    def stop(self) -> None:
        self.handle.stop()

    def close(self) -> None:
        if hasattr(self.handle, "close"):
            self.handle.close()


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

    stt: STTBase
    tts: TTSBase

    try:
        candidate_stt = create_stt()
        if not isinstance(candidate_stt, StubSTT):
            stt = candidate_stt
        else:
            from handlers.STTHandle import STTHandle

            stt = RealSTTAdapter(STTHandle(language="pl"))
            Logger.printInfo("Załadowano moduł STT (Faster-Whisper).")
    except Exception as exc:
        Logger.printWarning(f"Moduł STT niedostępny ({exc}) — używam StubSTT.")
        stt = StubSTT()

    try:
        candidate_tts = create_tts()
        if not isinstance(candidate_tts, StubTTS):
            tts = candidate_tts
        else:
            from handlers.TTSHandle import TTSHandle

            tts = RealTTSAdapter(TTSHandle(on_error=lambda err: Logger.printError(f"TTS: {err}")))
            Logger.printInfo("Załadowano moduł TTS (Piper).")
    except Exception as exc:
        Logger.printWarning(f"Moduł TTS niedostępny ({exc}) — używam StubTTS.")
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


def _render_thinking_block(thinking_text: str) -> None:
    """Czytelnie formatuje i zawija tekst myślenia modelu."""
    import shutil
    import textwrap

    term_width = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)
    wrap_width = max(term_width - 6, 40)
    bar_width = max(wrap_width + 2, 45)

    print(f"\n\033[38;5;135m╭── Proces myślenia {'─' * max(0, bar_width - 19)}╮\033[0m")
    for raw_line in thinking_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            print("\033[38;5;135m│\033[0m")
            continue
        wrapped = textwrap.wrap(line, width=wrap_width)
        for part in wrapped:
            print(f"\033[38;5;135m│\033[0m  \033[2;37m{part}\033[0m")
    print(f"\033[38;5;135m╰{'─' * (bar_width + 1)}╯\033[0m\n")


def _render_cli_banner(
    model_name: str, audio_enabled: bool, endpoint: str, tts_enabled: bool = False
) -> None:
    """Rysuje nowoczesny, responsywny banner startowy dla trybu CLI, pokazujący model, endpoint, katalog i status audio."""
    import shutil

    term_width = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)
    inner_width = max(term_width - 4, 52)

    cwd_str = str(Path.cwd())
    if len(cwd_str) > inner_width - 12:
        cwd_str = "..." + cwd_str[-(inner_width - 15):]

    def pad_line(text_raw: str, text_visible_len: int) -> str:
        spaces = max(0, inner_width - text_visible_len)
        return f"\033[38;5;208m│\033[0m  {text_raw}{' ' * spaces}\033[38;5;208m│\033[0m"

    title = " Jarvis CLI v0.1 "
    top_bar = "─" * 2 + title + "─" * max(0, inner_width - len(title))
    div_bar = "─" * (inner_width + 2)
    bottom_bar = "─" * (inner_width + 2)

    if not audio_enabled:
        audio_status = "OFF (--no-audio)"
        audio_color = "\033[31mOFF (--no-audio)\033[0m"
    else:
        tts_part = "TTS: ON" if tts_enabled else "TTS: OFF"
        audio_status = f"ON ({tts_part})"
        tts_col = "\033[32mON\033[0m" if tts_enabled else "\033[33mOFF\033[0m"
        audio_color = f"\033[32mON\033[0m (TTS: {tts_col})"

    print(f"\033[38;5;208m╭{top_bar}╮\033[0m")
    print(pad_line(f"Model:    \033[1m{model_name}\033[0m", 10 + len(model_name)))
    print(pad_line(f"Endpoint: \033[2m{endpoint}\033[0m", 10 + len(endpoint)))
    print(pad_line(f"Katalog:  \033[2m{cwd_str}\033[0m", 10 + len(cwd_str)))
    print(pad_line(f"Głos:     {audio_color}", 10 + len(audio_status)))
    print(f"\033[38;5;208m├{div_bar}┤\033[0m")

    line1_vis = "Komendy:  /voice (mikrofon)       /devices (mikrofony)"
    line1_sty = "Komendy:  \033[36m/voice\033[0m (mikrofon)       \033[36m/devices\033[0m (mikrofony)"
    line2_vis = "          /think (myślenie)       /think live (na żywo)"
    line2_sty = "          \033[36m/think\033[0m (myślenie)       \033[36m/think live\033[0m (na żywo)"
    line3_vis = "          /tts (głos ON/OFF)      /logs (logi)"
    line3_sty = "          \033[36m/tts\033[0m (głos ON/OFF)      \033[36m/logs\033[0m (logi)"
    line4_vis = "          /reset (nowy czat)      exit (wyjście)"
    line4_sty = "          \033[36m/reset\033[0m (nowy czat)      \033[36mexit\033[0m (wyjście)"

    print(pad_line(line1_sty, len(line1_vis)))
    print(pad_line(line2_sty, len(line2_vis)))
    print(pad_line(line3_sty, len(line3_vis)))
    print(pad_line(line4_sty, len(line4_vis)))

    print(f"\033[38;5;208m╰{bottom_bar}╯\033[0m\n")


def run_cli(runtime: AppRuntime) -> int:
    """Tryb CLI korzystający z tej samej pętli agenta co GUI."""
    Logger.VERBOSE = 1
    agent = runtime.agent_factory(runtime.settings)
    model_name = runtime.provider.model or runtime.settings.get("ollama_model") or "domyślny"

    _render_cli_banner(
        model_name,
        runtime.audio_enabled,
        runtime.provider.base_url,
        runtime.settings.get("tts_enabled", False),
    )

    show_thinking_live = False
    last_thinking = ""

    while True:
        try:
            # Claude-style prompt marker
            user_input = input("\033[1;38;5;208m❯\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[2mKończenie sesji Jarvis CLI...\033[0m")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "/exit", "/quit"):
            print("\033[2mDo widzenia!\033[0m")
            break
        if cmd == "/reset":
            agent.reset()
            last_thinking = ""
            print("\033[2m[Jarvis] Historia rozmowy wyczyszczona.\033[0m\n")
            continue
        if cmd == "/voice":
            if not runtime.audio_enabled:
                print("\033[31m[System] Funkcje audio są wyłączone flagą --no-audio.\033[0m\n")
                continue
            try:
                runtime.stt.start()
                input("\033[33m● Nagrywanie z mikrofonu... [Enter = stop]\033[0m")
                user_input = runtime.stt.stop().strip()
                print(f"\033[2mRozpoznano:\033[0m {user_input}")
                if not user_input:
                    continue
            except Exception as exc:
                Logger.printError(f"STT: {exc}")
                continue
        if cmd.startswith("/devices"):
            try:
                import sounddevice as sd

                parts = cmd.split()
                stt_handle = getattr(runtime.stt, "handle", None)

                if len(parts) > 1 and parts[1].isdigit():
                    dev_id = int(parts[1])
                    if stt_handle is not None and hasattr(stt_handle, "input_device"):
                        setattr(stt_handle, "input_device", dev_id)
                        print(f"\033[32m[STT] Ustawiono urządzenie wejściowe audio na ID {dev_id}.\033[0m\n")
                    else:
                        print("\033[33m[STT] Aktywny adapter audio nie obsługuje zmiany urządzenia.\033[0m\n")
                    continue

                devices = sd.query_devices()
                curr_dev = getattr(stt_handle, "input_device", None) if stt_handle is not None else None
                default_in = curr_dev if curr_dev is not None else sd.default.device[0]
                print("\n\033[2mDostępne mikrofony (wpisz '/devices <id>', aby zmienić):\033[0m")
                found = False
                for idx, dev in enumerate(devices):
                    if dev.get("max_input_channels", 0) > 0:
                        is_def = " \033[32m[AKTYWNY]\033[0m" if idx == default_in else ""
                        print(f"  \033[36m[{idx}]\033[0m {dev.get('name')}{is_def}")
                        found = True
                if not found:
                    print("  \033[31mNie wykryto żadnego mikrofonu w systemie.\033[0m")
                print()
            except Exception as exc:
                print(f"\033[31mBłąd odczytu urządzeń audio: {exc}\033[0m\n")
            continue

        if cmd.startswith("/tts"):
            if not runtime.audio_enabled:
                print("\033[31m[System] Funkcje audio są wyłączone flagą --no-audio.\033[0m\n")
                continue
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""

            if arg.lower() in ("test", "demo"):
                test_text = "Cześć! Synteza mowy Piper działa poprawnie."
                print(f"\033[2m[TTS] Testowanie syntezy mowy: '{test_text}'...\033[0m")
                try:
                    runtime.tts.speak(test_text)
                    print("\033[32m[TTS] Odtwarzanie zakończone pomyślnie.\033[0m\n")
                except Exception as exc:
                    print(f"\033[31m[TTS] Błąd syntezy mowy: {exc}\033[0m\n")
                continue

            if arg and arg.lower() not in ("on", "off"):
                print(f"\033[2m[TTS] Odtwarzanie tekstu: '{arg}'...\033[0m")
                try:
                    runtime.tts.speak(arg)
                    print("\033[32m[TTS] Odtwarzanie zakończone.\033[0m\n")
                except Exception as exc:
                    print(f"\033[31m[TTS] Błąd syntezy mowy: {exc}\033[0m\n")
                continue

            current_tts = bool(runtime.settings.get("tts_enabled", False))
            if arg.lower() == "on":
                new_tts = True
            elif arg.lower() == "off":
                new_tts = False
            else:
                new_tts = not current_tts

            runtime.settings_store.update({"tts_enabled": new_tts})
            state_str = "\033[32mWŁĄCZONA\033[0m" if new_tts else "\033[31mWYŁĄCZONA\033[0m"
            print(f"[Jarvis] Automatyczna synteza mowy (TTS): {state_str}.\n")
            continue

        if cmd == "/think":
            if last_thinking:
                _render_thinking_block(last_thinking)
            else:
                print("\033[2m[Jarvis] Ostatnia odpowiedź nie zawierała procesu myślenia.\033[0m\n")
            continue

        if cmd == "/think live":
            show_thinking_live = not show_thinking_live
            state = "WŁĄCZONY" if show_thinking_live else "WYŁĄCZONY"
            print(f"\033[2m[Jarvis] Podgląd myślenia na żywo: {state}.\033[0m\n")
            continue
        if cmd == "/logs":
            Logger.VERBOSE = 3 if Logger.VERBOSE < 3 else 1
            state = "WŁĄCZONE" if Logger.VERBOSE == 3 else "WYŁĄCZONE"
            print(f"\033[2m[Jarvis] Logi diagnostyczne: {state}.\033[0m\n")
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
                        print("\n\033[35m[Thinking]\033[0m ", end="", flush=True)
                        in_thinking = True
                    print(text, end="", flush=True)
                elif not in_thinking:
                    print("\r\033[2mJarvis myśli...\033[0m", end="", flush=True)
                    in_thinking = True

            elif event["type"] == "token":
                if in_thinking:
                    if show_thinking_live:
                        print("\n")
                    else:
                        print("\r" + " " * 30 + "\r", end="", flush=True)
                    in_thinking = False
                if not header_printed:
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
            print("\r" + " " * 30 + "\r\033[2m(brak odpowiedzi)\033[0m", end="", flush=True)

        last_thinking = "".join(thinking_parts).strip()
        if last_thinking and not show_thinking_live:
            print("\n\033[2m  (wpisz /think, aby zobaczyć proces myślenia)\033[0m\n")
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
