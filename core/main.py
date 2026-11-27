from __future__ import annotations

import argparse
import json
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
    STTBase,
    TTSBase,
    create_agent,
    create_stt,
    create_tts,
)

from core.Cli import (  
    _cli_approval,
    _cli_question,
    _cli_save_file_dialog,
    _cli_speak,
    _cli_tool_summary,
    _ensure_cli_provider_ready,
    _handle_cli_export,
    _render_cli_banner,
    _render_thinking_block,
    _start_cli_provider_watcher,
    run_cli,
)

from core.Attachments import AttachmentStore
from core.ChatStore import ChatStore, SettingsStore
from handlers.ToolHandle import ToolHandle
from tools.ToolBuilder import build_tools
from core.RuntimeConfig import RuntimeConfig
from core.ModelProvider import ModelProvider, create_provider
from core.AuditLog import AuditLog
from core.Interactions import InteractionBroker


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
    provider: ModelProvider
    stt: STTBase
    tts: TTSBase
    agent_factory: Callable[[dict], AgentBase]
    # #51: magazyn załączników (ID-based); WebGui binduje go z sesją agenta.
    attachment_store: AttachmentStore | None = None
    audio_enabled: bool = True
    config: RuntimeConfig | None = None
    provider_factory: Callable[[dict], ModelProvider] | None = None
    providers: list[Any] = field(default_factory=list)
    config_locked: bool = False
    _closed: bool = False
    mcp_clients: list[Any] = field(default_factory=list)
    interaction_broker: InteractionBroker | None = None
    audit_log: AuditLog | None = None
    provider_warning: str | None = None

    @property
    def settings(self) -> dict:
        return self.settings_store.get()

    def close(self) -> None:
        """Zamyka zasoby aplikacji: TTS, STT, MCP oraz sesje."""
        if self._closed:
            return
        self._closed = True
        for provider in {id(p): p for p in [self.provider, *self.providers]}.values():
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.providers.clear()
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

        if self.interaction_broker is not None:
            try:
                self.interaction_broker.close()
            except Exception:
                pass
        if self.audit_log is not None:
            try:
                self.audit_log.close()
            except Exception:
                pass

        if "core.WebGui" in sys.modules:
            try:
                sys.modules["core.WebGui"].sessions.clear()
            except Exception:
                pass


CONFIG_DESCRIPTIONS: dict[str, str] = {
    "ollama_endpoint": "Adres serwera modeli (np. http://localhost:11434/v1)",
    "ollama_model": "Nazwa modelu używanego przez agenta (np. qwen3.5:4b)",
    "system_prompt": "Edytowalne instrukcje systemowe agenta (stałe zasady narzędzi są dodawane automatycznie)",
    "max_chats": "Maksymalna liczba zapisanych rozmów (liczba całkowita, np. 50)",
    "max_age_days": "Maksymalny wiek zapisanych rozmów w dniach (np. 30)",
    "tts_enabled": "Automatyczne czytanie odpowiedzi na głos (true / false)",
    "stt_auto_send": "Automatyczne wysyłanie tekstu po nagraniu głosu (true / false)",
    "theme": "Motyw kolorystyczny GUI (dark / light)",
    "show_history": "Pokazywanie paska historii rozmów w GUI (true / false)",
    "tool_policies": "Nadpisania polityk narzędzi {nazwa: allow|ask|deny}",
    "tool_risk_policies": "Domyślne polityki per ryzyko {read|write|destructive|external: allow|ask|deny}",
}

DEFAULT_MCP_CONFIG_NAME = "mcp_servers.json"


def _protected_tool_paths(
    settings_store: SettingsStore,
    config: RuntimeConfig,
    mcp_config: str | None = None,
) -> list[Path]:
    """Files the agent must not modify even after an explicit approval."""
    mcp_path = Path(mcp_config).expanduser() if mcp_config else (
        settings_store.path.parent / DEFAULT_MCP_CONFIG_NAME
    )
    return [
        settings_store.path,
        mcp_path,
        config.data_dir / "audit",
        config.data_dir / "logs",
        config.data_dir / "chats",
        config.data_dir / "attachments",
    ]


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
    if isinstance(default_val, dict):
        try:
            parsed = json.loads(raw_val)
        except ValueError as exc:
            raise StartupError(f"Wartość dla '{key}' musi być obiektem JSON.") from exc
        if not isinstance(parsed, dict):
            raise StartupError(f"Wartość dla '{key}' musi być obiektem JSON.")
        return parsed
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


# 2b. MCP (opcjonalne; brak pakietu lub konfiguracji nie blokuje aplikacji)
def build_mcp(
    tool_handle: ToolHandle,
    settings_store: SettingsStore,
    config_path: str | None = None,
    enabled: bool = True,
):
    if not enabled:
        Logger.printInfo("Obsługa MCP wyłączona flagą --no-mcp.")
        return None

    try:
        from handlers.MCPHandle import MCPHandle
    except Exception as exc:
        Logger.printWarning(f"Pakiet MCP niedostępny ({exc}) — pomijam serwery MCP.")
        return None

    path = (
        Path(config_path).expanduser()
        if config_path
        else settings_store.path.parent / DEFAULT_MCP_CONFIG_NAME
    )
    if config_path and not path.is_file():
        Logger.printWarning(f"Nie znaleziono pliku konfiguracji MCP: {path}")
        return None
    mcp_handle = None
    try:
        mcp_handle = MCPHandle(config_path=path, tool_handle=tool_handle)
        mcp_handle.start()
        return mcp_handle
    except Exception as exc:
        Logger.printWarning(f"Nie udało się uruchomić MCP ({path}): {exc}")
        if mcp_handle is not None:
            try:
                mcp_handle.close()
            except Exception:
                pass
        return None


# 3. Provider / Model
def build_provider(
    settings: dict,
    tool_handle: ToolHandle,
    provider_name: str = "ollama",
    require_connection: bool = False,
) -> ModelProvider:
    provider_norm = (provider_name or "ollama").strip().lower()
    if provider_norm not in ("ollama", "openai", "local"):
        raise StartupError(f"Nieobsługiwany provider: '{provider_name}'.")

    try:
        config = RuntimeConfig.from_env(settings, provider_name=provider_norm)
    except ValueError as exc:
        raise StartupError(str(exc)) from exc
    endpoint = config.endpoint
    handler = create_provider(config, tool_handle)

    try:
        models = [m.id for m in handler.GetModels()]
    except Exception as exc:
        msg = "Brak połączenia z serwerem modeli."
        if require_connection:
            handler.close()
            raise StartupError(msg) from exc
        Logger.printWarning(msg)
        return handler

    if not models:
        msg = f"Brak dostępnych modeli na serwerze ({endpoint})."
        if require_connection:
            handler.close()
            raise StartupError(msg)
        Logger.printWarning(msg)
        return handler

    target_model = config.model or models[0]
    if not handler.SetModel(target_model):
        msg = f"Model '{target_model}' nie jest dostępny na serwerze ({endpoint})."
        if require_connection:
            handler.close()
            raise StartupError(msg)
        Logger.printWarning(msg)

    return handler


# 4. Audio (STT / TTS)
def build_audio(
    no_audio: bool = False, settings: dict | None = None
) -> tuple[STTBase, TTSBase]:
    if no_audio:
        Logger.printInfo("Flaga --no-audio: funkcje głosowe (STT/TTS) zostały zablokowane.")
        return DisabledSTT(), DisabledTTS()

    return create_stt(settings), create_tts(settings)


# 5. Spięcie zależności w jednym miejscu: config -> ToolHandle -> provider -> agent -> GUI/CLI
def build_runtime(args: argparse.Namespace) -> AppRuntime:
    settings_store, chat_store = build_config(args.config)
    settings = settings_store.get()

    config = RuntimeConfig.from_env(settings, provider_name=args.provider)
    Logger.configure_file_logging(config.data_dir / "logs" / "app.log")
    Logger.printInfo(
        f"Start runtime: data_dir={config.data_dir} workspace={config.workspace} "
        f"allowed_dirs={list(config.allowed_directories)}"
    )
    tool_handle = build_tools(
        workspace_dir=config.workspace,
        allowed_dirs=config.allowed_directories,
        policy_overrides=settings.get("tool_policies", {}),
        risk_policies=settings.get("tool_risk_policies", {}),
        protected_paths=_protected_tool_paths(
            settings_store, config, getattr(args, "mcp_config", None)
        ),
    )
    attachment_store = AttachmentStore(config.data_dir / "attachments")  # #51
    interaction_broker = InteractionBroker(timeout=config.approval_timeout)
    audit_log = AuditLog(config.data_dir / "audit" / "tool_audit.jsonl")
    Logger.printInfo(
        f"Logi: {Logger.log_file_path()} | audit: {audit_log.path}"
    )
    providers = []

    def provider_factory(current_settings):
        cfg = RuntimeConfig.from_env(current_settings, provider_name=args.provider)
        provider = create_provider(cfg, tool_handle)
        providers.append(provider)
        return provider

    provider = provider_factory(settings)
    mcp_handle = None
    provider_warning: str | None = None
    try:
        mcp_handle = build_mcp(tool_handle, settings_store,
                               config_path=args.mcp_config, enabled=not args.no_mcp)
        if args.cli or args.benchmark:
            try:
                names = [m.id for m in provider.GetModels()]
            except Exception as exc:
                msg = (
                    f"Brak połączenia z serwerem modeli ({config.endpoint}). "
                    "Uruchom Ollamę, aby wysyłać zapytania do modelu."
                )
                if args.benchmark:
                    raise StartupError(msg) from exc
                provider_warning = msg
                Logger.printWarning(msg)
            else:
                if not names:
                    msg = f"Brak dostępnych modeli na serwerze ({config.endpoint})."
                    if args.benchmark:
                        raise StartupError(msg)
                    provider_warning = msg
                    Logger.printWarning(msg)
                else:
                    target = config.model or names[0]
                    try:
                        selected = provider.SetModel(target)
                    except Exception as exc:
                        msg = f"Model '{target}' nie jest dostępny na serwerze ({config.endpoint})."
                        if args.benchmark:
                            raise StartupError(msg) from exc
                        provider_warning = msg
                        Logger.printWarning(msg)
                    else:
                        if not selected:
                            msg = f"Model '{target}' nie jest dostępny na serwerze ({config.endpoint})."
                            if args.benchmark:
                                raise StartupError(msg)
                            provider_warning = msg
                            Logger.printWarning(msg)
        stt, tts = build_audio(no_audio=args.no_audio, settings=settings)
    except Exception:
        for component in [provider, mcp_handle]:
            close = getattr(component, 'close', None)
            if callable(close):
                close()
        interaction_broker.close()
        audit_log.close()
        raise

    def agent_factory(current_settings=None):
        cfg = dict(current_settings or settings_store.get())
        resolved = RuntimeConfig.from_env(cfg, provider_name=args.provider)
        cfg.update(ollama_endpoint=resolved.endpoint, ollama_model=resolved.model)
        return create_agent(
            cfg,
            tool_handle=tool_handle,
            provider=provider_factory(cfg),
            interaction_broker=interaction_broker,
            audit_log=audit_log,
        )

    return AppRuntime(
        settings_store=settings_store, chat_store=chat_store, tool_handle=tool_handle,
        provider=provider, stt=stt, tts=tts, agent_factory=agent_factory,
        audio_enabled=not args.no_audio, config=config, provider_factory=provider_factory,
        providers=providers, mcp_clients=[mcp_handle] if mcp_handle is not None else [],
        attachment_store=attachment_store,
        interaction_broker=interaction_broker, audit_log=audit_log,
        provider_warning=provider_warning,
    )


def build_api_runtime(config=None, *, provider_factory=None, tool_factory=None):
    """Headless defaults: no audio, no MCP, no model/network calls on construction."""
    if config is None:
        initial = RuntimeConfig.from_env()
        config = RuntimeConfig.from_env(SettingsStore(initial.data_dir / "settings.json").get())
    Logger.configure_file_logging(config.data_dir / "logs" / "app.log")
    Logger.printInfo(
        f"Start API runtime: data_dir={config.data_dir} workspace={config.workspace} "
        f"allowed_dirs={list(config.allowed_directories)}"
    )
    settings_store = SettingsStore(config.data_dir / 'settings.json')
    chat_store = ChatStore(config.data_dir / 'chats')
    attachment_store = AttachmentStore(config.data_dir / 'attachments')  # #51
    settings = settings_store.get()
    tools = tool_factory() if tool_factory else build_tools(
        workspace_dir=config.workspace,
        allowed_dirs=config.allowed_directories,
        policy_overrides=settings.get("tool_policies", {}),
        risk_policies=settings.get("tool_risk_policies", {}),
        protected_paths=_protected_tool_paths(settings_store, config),
    )
    interaction_broker = InteractionBroker(timeout=config.approval_timeout)
    audit_log = AuditLog(config.data_dir / "audit" / "tool_audit.jsonl")
    Logger.printInfo(f"Logi: {Logger.log_file_path()} | audit: {audit_log.path}")
    make_provider = provider_factory or create_provider
    providers = []

    def factory(settings):
        # Explicit config/env owns backend configuration; no credentials enter SettingsStore.
        from dataclasses import replace
        model = config.model or settings.get('ollama_model', '')
        p = make_provider(replace(config, model=model), tools)
        providers.append(p)
        return p

    primary = factory(settings_store.get())
    try:
        stt, tts = build_audio(no_audio=True)
        def agent_factory(settings):
            settings = dict(settings, ollama_endpoint=config.endpoint,
                            ollama_model=config.model or settings.get('ollama_model', ''))
            return create_agent(
                settings,
                tool_handle=tools,
                provider=factory(settings),
                interaction_broker=interaction_broker,
                audit_log=audit_log,
            )
        return AppRuntime(settings_store=settings_store, chat_store=chat_store,
                          tool_handle=tools, provider=primary, stt=stt, tts=tts,
                          agent_factory=agent_factory, audio_enabled=False,
                          attachment_store=attachment_store,
                          config=config, provider_factory=factory, providers=providers, config_locked=True,
                          interaction_broker=interaction_broker, audit_log=audit_log)
    except Exception:
        primary.close()
        interaction_broker.close()
        audit_log.close()
        raise

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
        "  python main.py --no-mcp                           # uruchomienie bez serwerów MCP\n"
        "  python main.py --mcp-config config/mcp_servers.example.json  # własna konfiguracja MCP\n"
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
        "--no-mcp",
        action="store_true",
        help="Wyłącz ładowanie serwerów MCP.",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        metavar="PLIK",
        help=(
            "Ścieżka do pliku konfiguracji serwerów MCP. "
            "Domyślnie: <katalog ustawień>/mcp_servers.json."
        ),
    )
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

        start_gui(runtime=runtime)
        return 0

    except (StartupError, ValueError) as err:
        Logger.printError(str(err))
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
