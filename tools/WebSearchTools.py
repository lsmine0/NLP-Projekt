import re
from typing import Callable, Dict, Optional

import requests
from bs4 import BeautifulSoup, NavigableString
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from handlers.ToolHandle import ToolErrorText, ToolHandle


# --- Stałe konfiguracyjne ---
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10           # górny limit wyników, żeby model nie zapchał kontekstu
DEFAULT_MAX_CHARS = 4000
REQUEST_TIMEOUT = 10             # sekundy
MAX_DOWNLOAD_BYTES = 3_000_000   # ~3 MB – ochrona przed pobieraniem ogromnych plików

# Tagi, które nigdy nie zawierają treści czytelnej dla człowieka
ALWAYS_REMOVE_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "template",
    "button", "input", "select", "textarea",
]
# Tagi układu strony (menu, stopki itp.) – usuwane tylko poza główną treścią
LAYOUT_TAGS = ["nav", "footer", "header", "aside"]
# Nawigacja oznaczona przez ARIA oraz elementy MediaWiki (Wikipedia) spoza treści artykułu:
# lista języków, pasek narzędzi, ramki boczne, szablony nawigacyjne, przypisy i linki [edytuj]
NOISE_SELECTORS = ", ".join([
    "[role=navigation]",
    ".vector-dropdown", ".vector-page-toolbar",
    ".infobox", ".navbox", ".sidebar",
    "sup.reference", ".mw-editsection",
])
# Tagi blokowe zaczynają nową linię; tagi inline (np. <a>, <b>) nie mogą rozbijać zdania
BLOCK_TAGS = [
    "address", "article", "aside", "blockquote", "caption", "details", "dd", "div", "dl", "dt",
    "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
    "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table", "tr", "ul",
]


def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """
    Wyszukuje aktualne informacje w internecie za pomocą DuckDuckGo.

    :param query: Zapytanie wyszukiwania (słowa kluczowe).
    :param max_results: Maksymalna liczba wyników do zwrócenia (domyślnie 5, maks. 10).
    :return: Sformatowana lista wyników (tytuł, URL, opis) lub komunikat błędu.
    """
    if not query or not str(query).strip():
        return ToolErrorText("Błąd: Zapytanie wyszukiwania nie może być puste.")
    query = str(query).strip()

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return ToolErrorText(f"Błąd: Parametr max_results musi być liczbą całkowitą (otrzymano: {max_results!r}).")
    if max_results <= 0:
        max_results = DEFAULT_MAX_RESULTS
    max_results = min(max_results, MAX_RESULTS_LIMIT)

    try:
        results = DDGS().text(query, max_results=max_results)
    except RatelimitException:
        return ToolErrorText("Błąd: Przekroczono limit zapytań do wyszukiwarki DuckDuckGo (rate limit). Spróbuj ponownie za chwilę.")
    except TimeoutException:
        return ToolErrorText("Błąd: Przekroczono czas oczekiwania na odpowiedź wyszukiwarki DuckDuckGo (timeout).")
    except DDGSException as e:
        return ToolErrorText(f"Błąd wyszukiwarki DuckDuckGo: {e}")
    except Exception as e:
        return ToolErrorText(f"Nieoczekiwany błąd podczas wyszukiwania w sieci: {e}")

    if not results:
        return f"Brak wyników wyszukiwania dla zapytania: '{query}'."

    formatted_output = [f"Wyniki wyszukiwania dla '{query}':\n"]
    for idx, item in enumerate(results, start=1):
        formatted_output.append(
            f"{idx}. Tytuł: {item.get('title', 'Brak tytułu')}\n"
            f"   URL: {item.get('href', 'Brak URL')}\n"
            f"   Opis: {item.get('body', 'Brak opisu')}\n"
        )
    return "\n".join(formatted_output)


def _get_readable_text(root) -> str:
    """Zamienia drzewo HTML na tekst: nowe linie tylko po blokach, zdania pozostają w całości."""
    for text_node in root.find_all(string=True):
        # Pomijamy komentarze i deklaracje (podklasy NavigableString) oraz tekst preformatowany
        if type(text_node) is NavigableString and text_node.find_parent("pre") is None:
            text_node.replace_with(re.sub(r"\s+", " ", text_node))

    for tag in root.find_all("br"):
        tag.replace_with("\n")
    for tag in root.find_all(["td", "th"]):
        tag.insert_after("\t")
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    return "\n".join(line.strip() for line in root.get_text().splitlines() if line.strip())


def _extract_text(html: bytes, encoding: Optional[str]) -> str:
    """Wyciąga czytelny tekst z HTML-a, pomijając skrypty, style i elementy układu strony."""
    # Przekazujemy bajty – BeautifulSoup sam wykryje kodowanie z <meta charset>,
    # jeśli serwer nie podał go w nagłówku Content-Type.
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding)

    for tag in soup(ALWAYS_REMOVE_TAGS):
        tag.decompose()
    for tag in soup.select(NOISE_SELECTORS):
        tag.decompose()

    main_content = soup.find("article") or soup.find("main")
    if main_content is not None:
        # Wewnątrz artykułu zostawiamy <header> (często zawiera tytuł),
        # usuwamy tylko zagnieżdżone menu, boczne panele i stopki.
        for tag in main_content(["nav", "aside", "footer"]):
            tag.decompose()
    else:
        # Brak semantycznego znacznika – czyścimy elementy układu z całej strony.
        # Celowo NIE usuwamy <form>: np. strony ASP.NET opakowują w niego całą treść.
        for tag in soup(LAYOUT_TAGS):
            tag.decompose()
        main_content = soup.body or soup

    return _get_readable_text(main_content)


def fetch_webpage_content(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    Pobiera i oczyszcza tekst ze wskazanej strony WWW z nadmiarowego kodu HTML/JS.

    :param url: Pełny adres URL strony (zaczynający się od http:// lub https://).
    :param max_chars: Maksymalna liczba znaków zwracanego tekstu (domyślnie 4000).
                      Wartość <= 0 oznacza użycie wartości domyślnej.
    :return: Oczyszczony tekst ze strony lub czytelny komunikat błędu.
    """
    if not url or not str(url).strip().startswith(("http://", "https://")):
        return ToolErrorText("Błąd: Podano nieprawidłowy adres URL. Wymagany prefiks http:// lub https://.")
    url = str(url).strip()

    # Walidacja parametrów PRZED wykonaniem zapytania sieciowego
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        return ToolErrorText(f"Błąd: Parametr max_chars musi być liczbą całkowitą (otrzymano: {max_chars!r}).")
    if max_chars <= 0:
        max_chars = DEFAULT_MAX_CHARS

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }

    try:
        with requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "html" not in content_type and "text/plain" not in content_type:
                return ToolErrorText(
                    f"Błąd: Strona {url} nie jest dokumentem HTML/tekstowym "
                    f"(Content-Type: {content_type}). Nie można wyodrębnić tekstu."
                )

            # Pobieramy maksymalnie MAX_DOWNLOAD_BYTES, żeby nie zapchać pamięci
            chunks, downloaded = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                chunks.append(chunk)
                downloaded += len(chunk)
                if downloaded >= MAX_DOWNLOAD_BYTES:
                    break
            raw_html = b"".join(chunks)

            # Kodowanie bierzemy z nagłówka tylko wtedy, gdy serwer je jawnie podał
            # (inaczej requests domyślnie zakłada ISO-8859-1 i psuje polskie znaki).
            encoding = response.encoding if "charset=" in content_type else None

        if "text/plain" in content_type:
            cleaned_text = raw_html.decode(encoding or "utf-8", errors="replace").strip()
        else:
            cleaned_text = _extract_text(raw_html, encoding)

        if not cleaned_text:
            return f"Informacja: Strona {url} nie zawiera czytelnego tekstu (może wymagać renderowania JS)."

        if len(cleaned_text) > max_chars:
            cleaned_text = (
                cleaned_text[:max_chars]
                + "\n\n...[Tekst ucięty, aby nie przekroczyć limitu okna kontekstowego]..."
            )

        return f"Treść strony ({url}):\n\n{cleaned_text}"

    except requests.exceptions.Timeout:
        return ToolErrorText(f"Błąd: Przekroczono czas oczekiwania (timeout {REQUEST_TIMEOUT}s) podczas łączenia z {url}.")
    except requests.exceptions.ConnectionError:
        return ToolErrorText(f"Błąd: Brak połączenia z serwerem lub nie można rozwiązać nazwy hosta dla {url}.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "nieznany"
        return ToolErrorText(f"Błąd HTTP {status} podczas próby pobrania strony {url}.")
    except requests.exceptions.RequestException as e:
        return ToolErrorText(f"Błąd zapytania HTTP podczas pobierania strony {url}: {e}")
    except Exception as e:
        return ToolErrorText(f"Nieoczekiwany błąd podczas pobierania treści strony {url}: {e}")


def register_web_search_tools(tool_handle: ToolHandle, overwrite: bool = True) -> None:
    """
    Rejestruje definicje narzędzi internetowych w przekazanym obiekcie ToolHandle
    zgodnie z sygnaturą ToolHandle.create_tool().
    """
    tool_handle.create_tool(
        name="search_web",
        description="Wyszukuje aktualne informacje w internecie i zwraca listę wyników (tytuł, URL, krótki opis).",
        properties={
            "query": {
                "type": "string",
                "description": "Zapytanie wyszukiwania do wpisania w wyszukiwarkę (słowa kluczowe).",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maksymalna liczba wyników do zwrócenia (domyślnie {DEFAULT_MAX_RESULTS}, maks. {MAX_RESULTS_LIMIT}).",
            },
        },
        required=["query"],
        overwrite=overwrite,
    )

    tool_handle.create_tool(
        name="fetch_webpage_content",
        description="Pobiera i oczyszcza tekst ze wskazanej strony WWW z nadmiarowego kodu HTML/JS.",
        properties={
            "url": {
                "type": "string",
                "description": "Pełny adres URL strony WWW (musi zaczynać się od http:// lub https://).",
            },
            "max_chars": {
                "type": "integer",
                "description": f"Maksymalna liczba znaków do pobrania, chroniąca okno kontekstowe (domyślnie {DEFAULT_MAX_CHARS}).",
            },
        },
        required=["url"],
        overwrite=overwrite,
    )

    for name, function in AVAILABLE_WEB_TOOLS.items():
        tool_handle.register_function(name, function)


# Słownik pomocniczy do mapowania nazw narzędzi na funkcje wykonywalne w głównej pętli agenta
AVAILABLE_WEB_TOOLS: Dict[str, Callable] = {
    "search_web": search_web,
    "fetch_webpage_content": fetch_webpage_content,
}
