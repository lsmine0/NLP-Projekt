"""Złożenie promptu systemowego: część edytowalna + automatyczny kontekst.

Kontekst narzędziowy celowo jest minimalny (kilka linii), a nie katalogem
opisów: pełne opisy i parametry narzędzi żyją w rejestrze
(``handlers.ToolHandle``) i są wysyłane modelowi w listie ``tools``, więc
duplikowanie ich w promptcie zabierało ponad 2 tys. znaków stałego kontekstu.
Agent dowiaduje się szczegółów wtedy, gdy ich potrzebuje - przez meta-narzędzie
``list_tools`` (``tools/MetaTools.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from handlers.ToolHandle import ToolHandle  # noqa: F401 - typ rejestru narzędzi

LIST_TOOLS_NAME = "list_tools"

# %A zawsze zwraca angielskie nazwy dni, więc lista dni tygodnia jest własna.
_WEEKDAYS_PL = (
    "poniedziałek", "wtorek", "środę", "czwartek",
    "piątek", "sobota", "niedziela",
)

# Zasady, które muszą pozostać w stałej części prompta: bez nich lokalny model
# zapisuje wywołania jako JSON w treści albo twierdzi, że coś wykonano.
_TOOL_RULES = (
    "Wywołuj narzędzia mechanizmem tool_calls (function calling); nie zapisuj "
    "wywołań jako JSON ani jako bloków kodu w treści odpowiedzi.",
    "Nie twierdź, że narzędzie coś wykonało, dopóki nie dostaniesz jego wyniku; "
    "rzetelnie relacjonuj błędy i ograniczenia. Po wyniku narzędzia zawsze "
    "napisz odpowiedź użytkownikowi - nie kończ tury samym procesem myślenia.",
    "Jeśli wynik narzędzia ma success=false i result.denied=true, nie próbuj "
    "obejść odmowy innym narzędziem. Poinformuj użytkownika i zaproponuj "
    "alternatywę.",
)


def _tool_names(tool_handle: ToolHandle | None) -> set[str]:
    """Nazwy narzędzi z rejestru - bez kopiowania opisów do prompta."""
    if tool_handle is None:
        return set()
    names = getattr(tool_handle, "tool_names", None)
    if callable(names):
        return {str(name) for name in names()}
    collected: set[str] = set()
    for tool in tool_handle.get_tools_list():
        function = tool.get("function", {}) if isinstance(tool, Mapping) else {}
        name = function.get("name")
        if isinstance(name, str) and name:
            collected.add(name)
    return collected


def _format_offset(now: datetime) -> str:
    offset = now.utcoffset()
    if offset is None:
        return "UTC"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    hours, minutes = divmod(abs(total), 3600)
    return f"{sign}{hours:02d}:{minutes:02d}"


def build_time_context(now: datetime | None = None) -> str:
    """Linia z aktualną datą i godziną, wstrzykiwana w automatyczny kontekst."""
    local = now if isinstance(now, datetime) else datetime.now().astimezone()
    date_part = f"{local.strftime('%Y-%m-%d')} ({_WEEKDAYS_PL[local.weekday()]})"
    utc = local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return (
        f"Aktualna data i hora: {date_part}, {local.strftime('%H:%M')} czasu "
        f"lokalnego ({_format_offset(local)}); UTC: {utc}. Używaj tej daty, gdy "
        "użytkownik pyta o „dzisiaj”, „wczoraj” albo „teraz”. Danych zmiennych "
        "w czasie (ceny, kursy walut, pogoda, wiadomości) nie podawaj z pamięci "
        "- sprawdź je odpowiednim narzędziem."
    )


def build_tool_prompt(
    tool_handle: ToolHandle | None, config: Mapping[str, Any]
) -> str:
    """Kilka linii prompta narzędziowego - katalog opisów jest w listie tools.

    Prompt nie wylicza już narzędzi ani nie powtarza policyk: model widzi je w
    listie ``tools`` (nazwa + krótki opis), a szczegóły parametrów zwraca
    meta-narzędzie ``list_tools``, gdy agent ich faktycznie potrzebuje.
    """
    names = _tool_names(tool_handle)

    if not names:
        lines = [
            "Narzędzia: w tej sesji nie ma żadnego zarejestrowanego narzędzia, "
            "więc odpowiadaj bez wywołania tools."
        ]
    else:
        lines = [
            "Narzędzia (tools) są wysyłane z żądaniem w listie `tools`: widzisz "
            "w niej nazwę, krótki opis i strukturę parametrów, ale nie opisy "
            "parametrów.",
        ]
        if LIST_TOOLS_NAME in names:
            lines.append(
                f"Dysponujesz narzędziem `{LIST_TOOLS_NAME}()`: bez argumentów "
                "zwraca nazwy wszystkich dostępnych narzędzi, a z argumentem "
                "`tool_name` pełny opis i parametry jednego z nich. Wywołaj je, "
                "gdy nie znasz nazwy albo argumentów; nie zgaduj parametrów."
            )
        lines.extend(_TOOL_RULES)

    max_tool_calls = config.get("max_tool_calls") if isinstance(config, Mapping) else None
    if (
        isinstance(max_tool_calls, int)
        and not isinstance(max_tool_calls, bool)
        and max_tool_calls > 0
    ):
        lines.append(
            f"Maksymalnie {max_tool_calls} wywołań narzędzi w jednej odpowiedzi."
        )

    return "\n".join(lines)


def build_system_prompt(
    user_prompt: str | None,
    tool_prompt: str,
    config: Mapping[str, Any],
) -> str:
    """Złóż edytowalną część użytkownika ze stałych sekcji automatycznych."""
    if user_prompt is None:
        user_prompt = config.get("system_prompt", "") if isinstance(config, Mapping) else ""
    if not isinstance(user_prompt, str):
        raise TypeError("system_prompt must be a string.")

    sections = []
    if user_prompt.strip():
        sections.append(
            "Instrukcje użytkownika (część edytowalna):\n" + user_prompt.strip()
        )

    now_override = config.get("now") if isinstance(config, Mapping) else None
    automatic_parts = [build_time_context(now_override)]
    automatic_parts.append(
        "Jeśli użytkownik prosi o utworzenie lub narysowanie diagramu i nie "
        "wskazuje innego formatu, przygotuj go w Mermaid. Umieść składnię w "
        "kompletnym, zamkniętym bloku fenced rozpoczynającym się od ```mermaid; "
        "zwróć go bezpośrednio w odpowiedzi. Nie używaj narzędzi, nie twórz ani "
        "nie wyszukuj plików .mermaid, nie podawaj poleceń CLI ani skryptów. "
        "Nie twierdź, że utworzyłeś plik. Nie zastępuj diagramu kodem aplikacji "
        "ani samym opisem. Dobierz typ "
        "diagramu, np. flowchart TD, sequenceDiagram lub erDiagram. W flowchart "
        "używaj poprawnej składni krawędzi A --> B; etykietę zapisuj jako "
        "A -->|tekst etykiety| B i nie umieszczaj strzałek wewnątrz etykiety. "
        "W sequenceDiagram notatki zapisuj jako Note over A,B: tekst, "
        "Note right of A: tekst lub Note left of A: tekst; nie używaj samotnego "
        "note \"tekst\" bez wskazania uczestnika."
    )
    if tool_prompt and tool_prompt.strip():
        automatic_parts.append(tool_prompt.strip())

    sections.append(
        "Stałe zasady i kontekst runtime (część automatyczna; stosuj niezależnie "
        "od instrukcji użytkownika powyżej):\n" + "\n\n".join(automatic_parts)
    )
    return "\n\n".join(sections)
