import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ddgs.exceptions import RatelimitException, TimeoutException

from handlers.ToolHandle import ToolHandle
from tools.WebSearchTools import (
    AVAILABLE_WEB_TOOLS,
    fetch_webpage_content,
    register_web_search_tools,
    search_web,
)

POLSKIE_ZNAKI = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
TEST_URL_PL = "https://pl.wikipedia.org/wiki/Polska"
TEST_URL_EN = "https://en.wikipedia.org/wiki/Python_(programming_language)"

wyniki = []  # lista (nazwa_testu, czy_zaliczony, komunikat)


def test(nazwa):
    """Dekorator: uruchamia test, łapie błędy i zapisuje wynik, żeby jeden błąd nie przerywał reszty."""
    def dekorator(func):
        def wrapper():
            print("\n" + "=" * 65)
            print(f"TEST: {nazwa}")
            print("=" * 65)
            try:
                func()
                wyniki.append((nazwa, True, ""))
                print("✅ OK")
            except AssertionError as e:
                wyniki.append((nazwa, False, str(e)))
                print(f"❌ NIEZALICZONY: {e}")
            except Exception as e:
                wyniki.append((nazwa, False, f"{type(e).__name__}: {e}"))
                print(f"❌ WYJĄTEK: {type(e).__name__}: {e}")
        return wrapper
    return dekorator


def falszywa_odpowiedz(body: bytes, content_type: str) -> MagicMock:
    """Tworzy sztuczną odpowiedź requests.get (bez użycia sieci)."""
    r = MagicMock()
    r.headers = {"Content-Type": content_type}
    r.encoding = requests.utils.get_encoding_from_headers(r.headers)
    r.iter_content.return_value = [body]
    r.raise_for_status.return_value = None
    r.__enter__.return_value = r
    return r


# ---------------------------------------------------------------
# Rejestracja narzędzi
# ---------------------------------------------------------------

@test("Rejestracja w ToolHandle i get_tools_list()")
def test_rejestracja():
    th = ToolHandle()
    register_web_search_tools(th)

    tools_list = th.get_tools_list()
    assert len(tools_list) == 2, f"Oczekiwano 2 narzędzi, otrzymano {len(tools_list)}"
    assert th.get_tool("search_web") is not None, "Brak narzędzia 'search_web'"
    assert th.get_tool("fetch_webpage_content") is not None, "Brak narzędzia 'fetch_webpage_content'"
    assert set(AVAILABLE_WEB_TOOLS) == {"search_web", "fetch_webpage_content"}, \
        "AVAILABLE_WEB_TOOLS nie zgadza się z zarejestrowanymi narzędziami"

    print("Schemat JSON przekazywany do OllamaHandle:")
    print(th.to_json(indent=2))


# ---------------------------------------------------------------
# Testy z siecią (prawdziwe zapytania)
# ---------------------------------------------------------------

@test("Wyszukiwanie w sieci (search_web)")
def test_wyszukiwanie():
    wynik = search_web(query="Ollama tool calling", max_results=3)
    print(wynik)
    assert wynik.startswith("Wyniki wyszukiwania"), f"Brak wyników: {wynik[:200]}"
    assert wynik.count("URL:") <= 3, "Zwrócono więcej wyników niż max_results"
    assert "Tytuł:" in wynik and "URL:" in wynik and "Opis:" in wynik, "Brakuje pól w wynikach"


@test("Pobieranie polskiej strony (kodowanie + limit znaków)")
def test_pobieranie_strony():
    wynik = fetch_webpage_content(url=TEST_URL_PL, max_chars=500)
    print(wynik)
    assert wynik.startswith("Treść strony"), f"Nie udało się pobrać strony: {wynik[:200]}"
    # Sprawdzamy samą treść: prefiks „Treść strony” zawiera polskie znaki, więc nie może być brany pod uwagę
    tresc = wynik.split("\n\n", 1)[1]
    assert POLSKIE_ZNAKI & set(tresc), "Brak polskich znaków – prawdopodobnie błąd kodowania"
    assert "<script" not in wynik and "<div" not in wynik, "W wyniku zostały znaczniki HTML"
    assert "[Tekst ucięty" in wynik, "Tekst nie został ucięty mimo max_chars=500"


@test("Artykuł z Wikipedii mieści się w domyślnym limicie znaków")
def test_tresc_artykulu():
    wynik = fetch_webpage_content(url=TEST_URL_EN)
    print(wynik[:500])
    assert wynik.startswith("Treść strony"), f"Nie udało się pobrać strony: {wynik[:200]}"
    assert "Afrikaans" not in wynik, "W wyniku została lista języków"
    assert "high-level" in wynik, "Brak treści artykułu w domyślnym limicie znaków"


@test("Wywołanie pobierania strony przez ToolHandle.execute_tool")
def test_wywolanie_agenta():
    tools = ToolHandle()
    register_web_search_tools(tools)
    response = falszywa_odpowiedz(b"<article>Test pobierania</article>", "text/html")
    with patch("tools.WebSearchTools.requests.get", return_value=response):
        wynik = json.loads(tools.execute_tool(
            "fetch_webpage_content", {"url": TEST_URL_PL, "max_chars": 200}
        ))
    assert wynik["success"], wynik["error"]
    assert "Test pobierania" in wynik["result"], wynik


@test("Nieistniejąca domena")
def test_brak_hosta():
    wynik = fetch_webpage_content("https://domena-widmo.invalid")
    print(wynik)
    assert "Brak połączenia" in wynik, f"Nieoczekiwany komunikat: {wynik}"


# ---------------------------------------------------------------
# Testy bez sieci (walidacja i mocki)
# ---------------------------------------------------------------

@test("Walidacja danych wejściowych")
def test_walidacja():
    przypadki = {
        "puste zapytanie": search_web("   "),
        "zły max_results": search_web("test", max_results="abc"),
        "URL bez http": fetch_webpage_content("example.com"),
        "zły max_chars": fetch_webpage_content("https://example.com", max_chars="abc"),
    }
    for nazwa, wynik in przypadki.items():
        print(f"{nazwa}: {wynik}")
        assert wynik.startswith("Błąd"), f"'{nazwa}' nie zwróciło komunikatu błędu: {wynik}"


@test("Timeout podczas pobierania strony (mock)")
def test_timeout_strony():
    with patch("tools.WebSearchTools.requests.get", side_effect=requests.exceptions.Timeout):
        wynik = fetch_webpage_content("https://example.com")
    print(wynik)
    assert "Przekroczono czas" in wynik, f"Nieoczekiwany komunikat: {wynik}"


@test("Rate limit i timeout wyszukiwarki (mock)")
def test_bledy_wyszukiwarki():
    for wyjatek, oczekiwane in [(RatelimitException, "limit zapytań"), (TimeoutException, "czas oczekiwania")]:
        with patch("tools.WebSearchTools.DDGS") as ddgs_mock:
            ddgs_mock.return_value.text.side_effect = wyjatek("test")
            wynik = search_web("cokolwiek")
        print(f"{wyjatek.__name__}: {wynik}")
        assert oczekiwane in wynik, f"Zły komunikat dla {wyjatek.__name__}: {wynik}"


@test("Czyszczenie HTML: kodowanie, <header> w artykule, <form>, PDF (mock)")
def test_czyszczenie_html():
    # Strona bez charset w nagłówku HTTP – kodowanie tylko w <meta>
    html = '<html><head><meta charset="utf-8"></head><body><p>Zażółć gęślą jaźń</p></body></html>'
    with patch("tools.WebSearchTools.requests.get", return_value=falszywa_odpowiedz(html.encode(), "text/html")):
        wynik = fetch_webpage_content("https://example.com")
    assert "Zażółć gęślą jaźń" in wynik, f"Błąd kodowania: {wynik}"

    # Tytuł w <header> wewnątrz <article> ma zostać, menu ma zniknąć
    html = "<body><nav>MENU</nav><article><header><h1>Tytuł</h1></header><p>Treść</p></article></body>"
    with patch("tools.WebSearchTools.requests.get", return_value=falszywa_odpowiedz(html.encode(), "text/html; charset=utf-8")):
        wynik = fetch_webpage_content("https://example.com")
    assert "Tytuł" in wynik and "MENU" not in wynik, f"Złe czyszczenie artykułu: {wynik}"

    # Strona w stylu ASP.NET – cała treść w <form>
    html = "<body><form><div>Treść strony</div><footer>stopka</footer></form></body>"
    with patch("tools.WebSearchTools.requests.get", return_value=falszywa_odpowiedz(html.encode(), "text/html; charset=utf-8")):
        wynik = fetch_webpage_content("https://example.com")
    assert "Treść strony" in wynik and "stopka" not in wynik, f"Złe czyszczenie <form>: {wynik}"

    # Plik niebędący HTML-em
    with patch("tools.WebSearchTools.requests.get", return_value=falszywa_odpowiedz(b"%PDF-1.4", "application/pdf")):
        wynik = fetch_webpage_content("https://example.com/plik.pdf")
    assert wynik.startswith("Błąd") and "HTML" in wynik, f"PDF nie został odrzucony: {wynik}"
    print("Wszystkie przypadki czyszczenia HTML poprawne.")


@test("Czyszczenie HTML: Wikipedia i całe zdania (mock)")
def test_czyszczenie_wikipedii():
    html = """<body><main>
        <header><h1>Tytuł</h1><div class="vector-dropdown">Afrikaans Deutsch</div></header>
        <div class="vector-page-toolbar">Czytaj Edytuj</div>
        <table class="infobox"><tr><td>Ramka</td></tr></table>
        <h2>Historia<span class="mw-editsection">[edytuj]</span></h2>
        <p>Python jest
           <a href="/a">językiem</a> <b>wysokiego</b> poziomu.<sup class="reference">[1]</sup></p>
        <p>Zobacz <a class="reference internal" href="/b">dokumentację</a>.</p>
        <!-- komentarz -->
        <ul><li>Pierwszy</li><li>Drugi</li></ul>
        <table class="navbox"><tr><td>Szablon</td></tr></table>
    </main></body>"""
    with patch("tools.WebSearchTools.requests.get", return_value=falszywa_odpowiedz(html.encode(), "text/html; charset=utf-8")):
        wynik = fetch_webpage_content("https://example.com")
    print(wynik)

    tresc = wynik.split("\n\n", 1)[1]
    oczekiwane = "Tytuł\nHistoria\nPython jest językiem wysokiego poziomu.\nZobacz dokumentację.\nPierwszy\nDrugi"
    assert tresc == oczekiwane, f"Nieoczekiwana treść: {tresc!r}"


def run_tests():
    testy = [
        test_rejestracja,
        test_walidacja,
        test_timeout_strony,
        test_bledy_wyszukiwarki,
        test_czyszczenie_html,
        test_czyszczenie_wikipedii,
        test_wyszukiwanie,
        test_pobieranie_strony,
        test_tresc_artykulu,
        test_wywolanie_agenta,
        test_brak_hosta,
    ]
    for t in testy:
        t()

    zaliczone = sum(1 for _, ok, _ in wyniki if ok)
    print("\n" + "=" * 65)
    print(f"PODSUMOWANIE: {zaliczone}/{len(wyniki)} testów zaliczonych")
    print("=" * 65)
    for nazwa, ok, komunikat in wyniki:
        print(f"{'✅' if ok else '❌'} {nazwa}" + (f" — {komunikat}" if komunikat else ""))

    return zaliczone == len(wyniki)


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)