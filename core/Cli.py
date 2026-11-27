from __future__ import annotations

import contextlib
import json
import re
import shutil
import socket
import sys
import textwrap
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlsplit

from core import Logger
from core.AgentAdapter import AgentBase

if TYPE_CHECKING:
    from core.main import AppRuntime


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
    model_name: str,
    audio_enabled: bool,
    endpoint: str,
    tts_enabled: bool = False,
    provider_warning: str | None = None,
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

    if provider_warning:
        model_vis = f"{model_name} (offline)"
        model_sty = f"\033[1m{model_name}\033[0m \033[33m(offline)\033[0m"
    else:
        model_vis = model_name
        model_sty = f"\033[1m{model_name}\033[0m"

    print(f"\033[38;5;208m╭{top_bar}╮\033[0m")
    print(pad_line(f"Model:    {model_sty}", 10 + len(model_vis)))
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
    line4_vis = "          /chats (historia)       /load <id> (wznów)"
    line4_sty = "          \033[36m/chats\033[0m (historia)       \033[36m/load <id>\033[0m (wznów)"
    line5_vis = "          /export [txt|md]        /reset (nowy czat)"
    line5_sty = "          \033[36m/export [txt|md]\033[0m        \033[36m/reset\033[0m (nowy czat)"
    line6_vis = "          exit (wyjście)"
    line6_sty = "          \033[36mexit\033[0m (wyjście)"

    print(pad_line(line1_sty, len(line1_vis)))
    print(pad_line(line2_sty, len(line2_vis)))
    print(pad_line(line3_sty, len(line3_vis)))
    print(pad_line(line4_sty, len(line4_vis)))
    print(pad_line(line5_sty, len(line5_vis)))
    print(pad_line(line6_sty, len(line6_vis)))

    print(f"\033[38;5;208m╰{bottom_bar}╯\033[0m\n")
    if provider_warning:
        print(f"\033[33m[Uwaga] {provider_warning}\033[0m")
        print(
            "\033[2mTryb offline: działają komendy (/chats, /load, /export, /tts, /devices, "
            "/logs, /reset, exit). Po uruchomieniu Ollamy CLI połączy się automatycznie.\033[0m\n"
        )


def _cli_tool_summary(event: dict) -> str:
    risk = ", ".join(event.get("risk") or []) or "brak ryzyka"
    source = event.get("source") or "builtin"
    server = event.get("server")
    source_label = f"MCP/{server}" if source == "mcp" and server else source
    return f"{event.get('tool')} [{source_label}] ryzyko: {risk}"


def _cli_approval(runtime: AppRuntime, event: dict) -> str:
    print("\n\033[33m╭─ Zgoda na narzędzie ─────────────────────────╮\033[0m")
    print(f"\033[33m│\033[0m {_cli_tool_summary(event)}")
    args = event.get("arguments") or {}
    if args:
        import json

        for line in json.dumps(args, ensure_ascii=False, indent=2).splitlines():
            print(f"\033[2m│ {line}\033[0m")
    print("\033[33m╰──────────────────────────────────────────────╯\033[0m")
    while True:
        try:
            choice = input("Zgoda: [1/a] raz  [2/s] w sesji  [3/d] odmów (domyślnie 3): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "3"
        decision = {
            "1": "allow_once", "a": "allow_once",
            "2": "allow_session", "s": "allow_session",
            "3": "deny", "d": "deny",
        }.get(choice, "deny")
        broker = runtime.interaction_broker
        if broker is None:
            return decision
        result = broker.resolve_approval(event["id"], decision)
        if result.ok:
            state = {"allow_once": "zezwolono raz", "allow_session": "zezwolono w sesji", "deny": "odmówiono"}[decision]
            print(f"\033[2m[{state}]\033[0m")
            return decision
        print(f"\033[31mNie można rozstrzygnąć zgody: {result.code}\033[0m")
        if result.code in ("expired", "cancelled", "not_found", "already_resolved"):
            return decision


def _cli_question(runtime: AppRuntime, event: dict) -> None:
    print("\n\033[36m╭─ Pytanie od agenta ──────────────────────────╮\033[0m")
    header = (event.get("header") or "").strip()
    if header:
        print(f"\033[36m│\033[0m \033[1m{header}\033[0m")
    print(f"\033[36m│\033[0m {event.get('question')}")
    options = list(event.get("options") or [])
    for index, option in enumerate(options, start=1):
        print(f"\033[36m│\033[0m   [{index}] {option}")
    print("\033[36m╰──────────────────────────────────────────────╯\033[0m")
    try:
        answer = input("Odpowiedź (numer opcji, własny tekst lub Enter = anuluj): ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    broker = runtime.interaction_broker
    if broker is None:
        return
    if not answer:
        payload = {"cancelled": True}
    elif answer.isdigit() and 1 <= int(answer) <= len(options):
        selected = options[int(answer) - 1]
        payload = {"answer": selected, "option": selected}
    elif event.get("allow_custom", True) is False:
        print("\033[31mTo pytanie pozwala tylko na wybór jednej z opcji.\033[0m")
        payload = {"cancelled": True}
    else:
        payload = {"answer": answer}
    result = broker.resolve_question(event["id"], payload)
    if not result.ok:
        print(f"\033[31mNie można przekazać odpowiedzi: {result.code}\033[0m")


_CLI_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)


def _cli_speak(
    runtime: AppRuntime,
    text: str,
    status_message: str = "[TTS] Odtwarzanie odpowiedzi... (Ctrl+C aby przerwać)",
) -> bool:
    print(f"\033[2m{status_message}\033[0m", flush=True)
    try:
        runtime.tts.speak(text)
        return True
    except KeyboardInterrupt:
        try:
            runtime.tts.stop()
        except Exception:
            pass
        print("\n\033[33m[TTS] Odtwarzanie przerwane.\033[0m\n")
        return False
    except Exception as exc:
        Logger.printError(f"TTS: {exc}")
        return False


def _bind_cli_chat(agent: AgentBase, runtime: AppRuntime, chat_id: str | None) -> None:
    attachment_store = getattr(runtime, "attachment_store", None)
    binder = getattr(agent, "bind_attachments", None)
    if attachment_store is not None and callable(binder):
        try:
            binder(attachment_store, chat_id)
            return
        except Exception:
            pass
    flow = getattr(agent, "flow", None)
    if flow is not None:
        scope = dict(getattr(flow, "session_scope", None) or {})
        scope["chat_id"] = chat_id
        flow.session_scope = scope


def _cleanup_cli_chats(runtime: AppRuntime) -> None:
    cfg = runtime.settings
    runtime.chat_store.cleanup(
        int(cfg.get("max_chats", 0) or 0),
        int(cfg.get("max_age_days", 0) or 0),
    )


def _resolve_chat_id(chats: list[dict], selector: str) -> tuple[str | None, str | None]:
    token = (selector or "").strip()
    if not token:
        return None, "Podaj ID lub numer czatu (użyj /chats, aby zobaczyć listę)."
    for chat in chats:
        if chat.get("id") == token:
            return token, None
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(chats):
            return str(chats[idx]["id"]), None
    matches = [chat["id"] for chat in chats if str(chat.get("id", "")).startswith(token)]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, f"Niejednoznaczny prefiks ID '{token}' ({len(matches)} dopasowania)."
    return None, f"Nie znaleziono czatu '{token}'."


def _extract_last_thinking(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        matches = _CLI_THINK_RE.findall(msg.get("content", "") or "")
        if matches:
            return matches[-1].strip()
    return ""


def _cli_save_file_dialog(
    filename: str, fmt: str, initial_dir: Path
) -> tuple[bool, Path | None]:
    if not sys.stdin.isatty():
        return False, None
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return False, None

    default_ext = ".txt" if fmt == "txt" else ".md"
    if fmt == "txt":
        filetypes = [("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")]
    else:
        filetypes = [("Pliki Markdown", "*.md"), ("Wszystkie pliki", "*.*")]

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Eksportuj rozmowę",
            initialdir=str(initial_dir),
            initialfile=filename,
            defaultextension=default_ext,
            filetypes=filetypes,
        )
    except Exception:
        return False, None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    if not selected:
        return True, None
    return True, Path(str(selected))


def _handle_cli_export(
    runtime: AppRuntime, current_chat_id: str | None, args_text: str
) -> None:
    if not current_chat_id:
        print(
            "\033[33m[Jarvis] Brak aktywnej zapisanej rozmowy. "
            "Wyślij wiadomość lub wczytaj czat komendą /load <nr lub id>.\033[0m\n"
        )
        return

    trimmed = args_text.strip()
    if not trimmed:
        try:
            fmt_raw = input(
                "Format eksportu [1/md] Markdown  [2/txt] TXT (Enter = md, 'anuluj' = przerwij): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[33m[Jarvis] Anulowano eksport.\033[0m\n")
            return
        if fmt_raw in ("anuluj", "cancel", "q"):
            print("\033[33m[Jarvis] Anulowano eksport.\033[0m\n")
            return
        fmt = {"": "md", "1": "md", "md": "md", "2": "txt", "txt": "txt"}.get(fmt_raw, fmt_raw)
        output_arg = ""
    else:
        parts = trimmed.split(maxsplit=1)
        fmt = parts[0].strip().lower()
        output_arg = parts[1].strip() if len(parts) > 1 else ""

    try:
        exported = runtime.chat_store.export_chat(
            current_chat_id,
            fmt,
            attachment_store=getattr(runtime, "attachment_store", None),
        )
    except ValueError as exc:
        print(f"\033[31m[Jarvis] {exc}\033[0m\n")
        return
    except KeyError:
        print(f"\033[31m[Jarvis] Nie znaleziono czatu '{current_chat_id}'.\033[0m\n")
        return

    downloads = Path.home() / "Downloads"
    default_dir = downloads if downloads.is_dir() else Path.cwd()
    default_target = default_dir / exported.filename

    target: Path
    if not output_arg:
        dialog_used, dialog_target = _cli_save_file_dialog(
            exported.filename, exported.format, default_dir
        )
        if dialog_used:
            if dialog_target is None:
                print("\033[33m[Jarvis] Anulowano eksport.\033[0m\n")
                return
            target = dialog_target
        else:
            try:
                chosen = input(
                    f"Gdzie zapisać plik? [Enter = {default_target}, 'anuluj' = przerwij]: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\033[33m[Jarvis] Anulowano eksport.\033[0m\n")
                return
            if chosen.lower() in ("anuluj", "cancel", "q"):
                print("\033[33m[Jarvis] Anulowano eksport.\033[0m\n")
                return
            output_arg = chosen
            if output_arg:
                cleaned_arg = output_arg.strip().strip('"').strip("'")
                raw_ends_with_sep = cleaned_arg.endswith(("/", "\\"))
                candidate = Path(cleaned_arg).expanduser()
                if (candidate.exists() and candidate.is_dir()) or raw_ends_with_sep:
                    target = candidate / exported.filename
                else:
                    target = candidate
            else:
                target = default_target
    else:
        cleaned_arg = output_arg.strip().strip('"').strip("'")
        raw_ends_with_sep = cleaned_arg.endswith(("/", "\\"))
        candidate = Path(cleaned_arg).expanduser()
        if (candidate.exists() and candidate.is_dir()) or raw_ends_with_sep:
            target = candidate / exported.filename
        else:
            target = candidate

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(exported.content, encoding="utf-8")
    except OSError as exc:
        print(f"\033[31m[Jarvis] Nie udało się zapisać pliku eksportu: {exc}\033[0m\n")
        return

    print(f"\033[32m[Jarvis] Wyeksportowano rozmowę do: {target}\033[0m\n")


def _cli_prompt_text(offline: bool) -> str:
    if offline:
        return "\033[33m(offline)\033[0m \033[1;38;5;208m❯\033[0m "
    return "\033[1;38;5;208m❯\033[0m "


def _is_endpoint_reachable(endpoint: str, timeout: float = 0.5) -> bool:
    parsed = urlsplit(endpoint)
    host = parsed.hostname
    if not host:
        return True
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_cli_provider_ready(
    runtime: AppRuntime,
    agent: AgentBase,
    *,
    provider_lock: threading.Lock | None = None,
    announce: bool = True,
) -> tuple[bool, str | None]:
    lock_ctx: Any = provider_lock if provider_lock is not None else contextlib.nullcontext()
    with lock_ctx:
        if not getattr(runtime, "provider_warning", None) and getattr(runtime.provider, "model", None):
            return True, None

        endpoint = getattr(runtime.provider, "base_url", None) or runtime.settings.get("ollama_endpoint", "")
        try:
            names = [m.id for m in runtime.provider.GetModels()]
        except Exception:
            msg = (
                f"Brak połączenia z serwerem modeli ({endpoint}). "
                "Uruchom Ollamę, aby wysyłać zapytania do modelu."
            )
            runtime.provider_warning = msg
            return False, msg

        if not names:
            msg = f"Brak dostępnych modeli na serwerze ({endpoint})."
            runtime.provider_warning = msg
            return False, msg

        cfg = runtime.config
        cfg_model = cfg.model if cfg is not None else ""
        target = cfg_model or runtime.settings.get("ollama_model") or names[0]
        try:
            selected = runtime.provider.SetModel(target)
        except Exception:
            selected = False
        if not selected:
            msg = f"Model '{target}' nie jest dostępny na serwerze ({endpoint})."
            runtime.provider_warning = msg
            return False, msg

        runtime.provider_warning = None
        if hasattr(agent, "model") and not getattr(agent, "model", ""):
            setattr(agent, "model", runtime.provider.model)
        if announce:
            print(f"\033[32m[System] Połączono z serwerem modeli ({runtime.provider.model}).\033[0m")
        return True, None


def _start_cli_provider_watcher(
    runtime: AppRuntime,
    agent: AgentBase,
    provider_lock: threading.Lock,
    stop_event: threading.Event,
    waiting_for_input: threading.Event,
    *,
    interval: float = 2.0,
    probe: Callable[[], bool] | None = None,
) -> threading.Thread:
    endpoint = getattr(runtime.provider, "base_url", None) or runtime.settings.get("ollama_endpoint", "")
    check_reachable = probe or (lambda: _is_endpoint_reachable(endpoint))

    def watch() -> None:
        wait_time = interval
        while not stop_event.wait(wait_time):
            if not getattr(runtime, "provider_warning", None) and getattr(runtime.provider, "model", None):
                return
            if not check_reachable():
                wait_time = interval
                continue
            ok, _ = _ensure_cli_provider_ready(
                runtime,
                agent,
                provider_lock=provider_lock,
                announce=False,
            )
            if ok:
                msg = f"\033[32m[System] Połączono z serwerem modeli ({runtime.provider.model}).\033[0m"
                if waiting_for_input.is_set():
                    print(f"\r\033[K{msg}\n{_cli_prompt_text(False)}", end="", flush=True)
                else:
                    print(f"\n{msg}", flush=True)
                return
            wait_time = min(wait_time * 1.5, 10.0)

    thread = threading.Thread(target=watch, name="cli-provider-watch", daemon=True)
    thread.start()
    return thread


def run_cli(runtime: AppRuntime) -> int:
    """Tryb CLI korzystający z tej samej pętli agenta co GUI."""
    Logger.VERBOSE = 1
    agent = runtime.agent_factory(runtime.settings)
    model_name = runtime.provider.model or runtime.settings.get("ollama_model") or "domyślny"
    provider_warning = getattr(runtime, "provider_warning", None)

    _render_cli_banner(
        model_name,
        runtime.audio_enabled,
        runtime.provider.base_url,
        runtime.settings.get("tts_enabled", False),
        provider_warning=provider_warning,
    )

    show_thinking_live = False
    last_thinking = ""
    current_chat_id: str | None = None
    provider_lock = threading.Lock()
    watcher_stop = threading.Event()
    waiting_for_input = threading.Event()

    if provider_warning and sys.stdin.isatty():
        _start_cli_provider_watcher(
            runtime,
            agent,
            provider_lock,
            watcher_stop,
            waiting_for_input,
        )

    try:
        while True:
            prompt = _cli_prompt_text(bool(getattr(runtime, "provider_warning", None)))
            waiting_for_input.set()
            try:
                # Claude-style prompt marker
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\033[2mKończenie sesji Jarvis CLI...\033[0m")
                break
            finally:
                waiting_for_input.clear()

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("exit", "quit", "/exit", "/quit"):
                print("\033[2mDo widzenia!\033[0m")
                break
            if cmd == "/chats":
                chats = runtime.chat_store.list()
                if not chats:
                    print("\033[2m[Jarvis] Brak zapisanych rozmów.\033[0m\n")
                    continue
                print("\n\033[2mZapisane rozmowy (użyj '/load <nr lub id>', aby wznowić):\033[0m")
                for idx, chat in enumerate(chats, start=1):
                    active_tag = " \033[32m[AKTYWNY]\033[0m" if chat.get("id") == current_chat_id else ""
                    short_id = str(chat.get("id", ""))[:8]
                    updated = str(chat.get("updated_at", ""))[:19].replace("T", " ")
                    title = chat.get("title") or "Nowa rozmowa"
                    print(
                        f"  \033[36m[{idx}]\033[0m {short_id}  \033[2m{updated}\033[0m  "
                        f"{title}{active_tag}"
                    )
                print()
                continue
            if cmd == "/load" or cmd.startswith("/load "):
                parts = user_input.split(maxsplit=1)
                selector = parts[1].strip() if len(parts) > 1 else ""
                if not selector:
                    print("\033[33m[Jarvis] Użycie: /load <nr lub id> (zobacz /chats).\033[0m\n")
                    continue
                chats = runtime.chat_store.list()
                target_id, err_msg = _resolve_chat_id(chats, selector)
                if not target_id:
                    print(f"\033[31m[Jarvis] {err_msg}\033[0m\n")
                    continue
                try:
                    chat = runtime.chat_store.get(target_id)
                except KeyError:
                    print(f"\033[31m[Jarvis] Nie znaleziono czatu '{target_id}'.\033[0m\n")
                    continue
                messages = chat.get("messages", [])
                agent.load_history(messages)
                current_chat_id = str(chat["id"])
                _bind_cli_chat(agent, runtime, current_chat_id)
                if runtime.interaction_broker is not None:
                    runtime.interaction_broker.clear_session_allows()
                last_thinking = _extract_last_thinking(messages)
                print(
                    f"\033[32m[Jarvis] Wczytano rozmowę '{chat.get('title')}' "
                    f"({current_chat_id[:8]}, {len(messages)} wiadomości).\033[0m"
                )
                for msg in messages[-4:]:
                    role_label = "Ty" if msg.get("role") == "user" else "Jarvis"
                    preview = _CLI_THINK_RE.sub("", msg.get("content", "") or "").strip()
                    preview = " ".join(preview.split())
                    if len(preview) > 100:
                        preview = preview[:97].rstrip() + "..."
                    print(f"  \033[2m{role_label}: {preview or '(pusta)'}\033[0m")
                print()
                continue
            if cmd == "/reset":
                agent.reset()
                current_chat_id = None
                _bind_cli_chat(agent, runtime, None)
                if runtime.interaction_broker is not None:
                    runtime.interaction_broker.clear_session_allows()
                last_thinking = ""
                print("\033[2m[Jarvis] Historia rozmowy wyczyszczona.\033[0m\n")
                continue
            if cmd == "/export" or cmd.startswith("/export "):
                parts = user_input.split(maxsplit=1)
                _handle_cli_export(runtime, current_chat_id, parts[1] if len(parts) > 1 else "")
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
                    if _cli_speak(
                        runtime,
                        test_text,
                        f"[TTS] Testowanie syntezy mowy: '{test_text}'... (Ctrl+C aby przerwać)",
                    ):
                        print("\033[32m[TTS] Odtwarzanie zakończone pomyślnie.\033[0m\n")
                    continue

                if arg and arg.lower() not in ("on", "off"):
                    if _cli_speak(
                        runtime,
                        arg,
                        f"[TTS] Odtwarzanie tekstu: '{arg}'... (Ctrl+C aby przerwać)",
                    ):
                        print("\033[32m[TTS] Odtwarzanie zakończone.\033[0m\n")
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

            ready, ready_msg = _ensure_cli_provider_ready(
                runtime, agent, provider_lock=provider_lock
            )
            if not ready:
                print(f"\033[33m[System] {ready_msg}\033[0m")
                print(
                    "\033[2mDostępne w trybie offline: /chats, /load <id>, /export, "
                    "/tts, /devices, /logs, /reset, exit.\033[0m\n"
                )
                continue

            if current_chat_id is None:
                current_chat_id = str(runtime.chat_store.create()["id"])
                _cleanup_cli_chats(runtime)
                _bind_cli_chat(agent, runtime, current_chat_id)
            runtime.chat_store.append(current_chat_id, "user", user_input)

            thinking_parts: list[str] = []
            reply_parts: list[str] = []
            in_thinking = False
            header_printed = False
            interrupted = False

            try:
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

                    elif event["type"] == "approval_required":
                        _cli_approval(runtime, event)
                    elif event["type"] == "user_input_required":
                        _cli_question(runtime, event)
                    elif event["type"] in ("approval_pending", "question_pending"):
                        continue
                    elif event["type"] == "warning":
                        Logger.printWarning(event["message"])
                    elif event["type"] == "answer_reset":
                        reply_parts.clear()
                        thinking_parts.clear()
                        header_printed = False
                        in_thinking = False
                        print("\n\033[33m[Retry]\033[0m Model opisał wywołanie narzędzia w treści — ponawiam z tool_calls.")
                    elif event["type"] == "tool_start":
                        print(f"\n\033[2m[tool] {_cli_tool_summary(event)}\033[0m")
                    elif event["type"] == "tool_result":
                        status = event.get("status", "ok")
                        color = "\033[32m" if status == "ok" else "\033[31m"
                        print(f"\033[2m[tool] {event.get('tool')}: {color}{status}\033[0m")
                    elif event["type"] == "error":
                        if in_thinking:
                            print("\r" + " " * 30 + "\r", end="", flush=True)
                            in_thinking = False
                        print()
                        Logger.printError(event["message"])
            except KeyboardInterrupt:
                interrupted = True
                flow = getattr(agent, "flow", None)
                broker = getattr(runtime, "interaction_broker", None)
                if broker is not None and flow is not None and getattr(flow, "run_id", None):
                    broker.cancel_run(flow.run_id)
                print("\n\033[33m[Jarvis] Generowanie przerwane.\033[0m")

            if in_thinking and not header_printed:
                print("\r" + " " * 30 + "\r\033[2m(brak odpowiedzi)\033[0m", end="", flush=True)

            last_thinking = "".join(thinking_parts).strip()
            if last_thinking and not show_thinking_live:
                print("\n\033[2m  (wpisz /think, aby zobaczyć proces myślenia)\033[0m\n")
            else:
                print("\n")

            full_reply = "".join(reply_parts).strip()
            stored_reply = "".join(reply_parts)
            if last_thinking and "<think>" not in stored_reply:
                stored_reply = f"<think>{''.join(thinking_parts)}</think>\n{stored_reply}"
            if interrupted:
                stored_reply = f"{stored_reply}\n<stopped/>".strip()
            if stored_reply or interrupted:
                runtime.chat_store.append(current_chat_id, "assistant", stored_reply)

            if full_reply and not interrupted and runtime.audio_enabled and runtime.settings.get("tts_enabled"):
                _cli_speak(runtime, full_reply)
    finally:
        watcher_stop.set()

    return 0

