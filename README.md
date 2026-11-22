# NLP-Projekt
Projekt z Przetwarzanie języka naturalnego

## GUI: Markdown i diagramy

GUI używa lokalnie bundlowanych bibliotek Marked, DOMPurify i Mermaid, więc
renderowanie Markdown i diagramów działa bez połączenia z CDN. Zamknięte bloki
fenced `mermaid` są renderowane jako diagramy; każdy blok ma przełącznik widoku
diagramu i kodu źródłowego. Niepoprawny diagram pozostaje dostępny jako kod z
komunikatem o błędzie, a renderowany Markdown i SVG są sanitizowane. Gdy
użytkownik prosi o diagram bez wskazania innego formatu, model otrzymuje
instrukcję, by odpowiedzieć blokiem Mermaid.

## API bez okna i audio (zadanie #55)

Wymagany Python 3.10 lub nowszy (sprawdzono 3.12). Tryb headless nie wymaga pywebview, mikrofonu, modeli STT/TTS ani MCP. Dotychczasowy desktop/CLI nadal korzysta z `core/main.py` i pełnego `requirements.txt`.

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
python -m uvicorn core.api:create_app --factory --host 127.0.0.1 --port 8000
```

Polecenia wykonuj z głównego folderu repozytorium. Import modułu i utworzenie aplikacji nie uruchamiają modelu, audio, okna ani serwerów MCP. `lifespan` buduje `AppRuntime` i zamyka jego zasoby przy wyłączeniu API. Dostępność modelu jest sprawdzana przez readiness lub przy zapytaniu do modelu, a nie przy imporcie. Każda instancja aplikacji ma własne sesje, magazyn załączników i runtime.

### Konfiguracja procesu

| Zmienna | Znaczenie / domyślna wartość |
|---|---|
| `NLP_ENDPOINT` | Adres API modelu, np. `http://127.0.0.1:11434/v1` |
| `NLP_MODEL` | Nazwa modelu; pusty wybiera pierwszy dostępny |
| `NLP_PROVIDER` | `ollama` (domyślnie), `local` lub `openai`; obecny adapter korzysta z OllamaHandler i API zgodnego z OpenAI |
| `OPENAI_API_KEY` / `OLLAMA_API_KEY` | Klucz wyłącznie w środowisku; pierwszeństwo ma `OPENAI_API_KEY`; dla lokalnej Ollamy bez klucza używane jest `ollam` |
| `NLP_REQUEST_TIMEOUT` | Dodatni timeout pojedynczej operacji sieciowej, domyślnie 60 s; brak ponowień klienta SDK |
| `NLP_WORKSPACE` | Główny katalog dla narzędzi plikowych i kodowania, domyślnie bieżący katalog |
| `NLP_ALLOWED_DIRS` | Dodatkowe dozwolone katalogi robocze, oddzielone separatorem systemowym (`:` / `;`) |
| `NLP_APPROVAL_TIMEOUT` | Timeout oczekiwania na zgodę/pytanie użytkownika w sekundach, domyślnie 300 |
| `CHAT_DATA_DIR` | Katalog ustawień, czatów, logów, audytu i załączników, domyślnie `data/` w repozytorium |
| `NLP_LOG_FILE` | Opcjonalna ścieżka pliku logu; domyślnie `$CHAT_DATA_DIR/logs/app.log` |

Dla endpointu/modelu headless zachowuje odczyt istniejącego `settings.json`, jeśli nie ustawiono `NLP_ENDPOINT`/`NLP_MODEL`. Obsługiwane są też dotychczasowe `OLLAMA_ENDPOINT`/`OLLAMA_MODEL` jako wartości domyślne ustawień. Klucza ani timeoutu/workspace nie zapisuje się w `settings.json`. W trybie headless endpoint i model są ustalane przy starcie procesu; zmiana przez `/api/settings` jest odrzucana (pozostałe ustawienia GUI nadal można zapisywać). Endpoint nie może zawierać loginu, hasła ani parametrów query. Klucz podawaj nagłówkiem przez konfigurację providera, nigdy w adresie URL.

Przykład macOS/Linux:

```sh
export NLP_ENDPOINT=http://127.0.0.1:11434/v1
export NLP_MODEL=qwen3:4b
export NLP_REQUEST_TIMEOUT=30
export NLP_WORKSPACE="$PWD"
python -m uvicorn core.api:create_app --factory --host 127.0.0.1 --port 8000
```

Workspace nie zmienia bieżącego katalogu procesu. Narzędzia plikowe sprawdzają, czy rozstrzygnięta ścieżka pozostaje w workspace; narzędzia kodowania otrzymują workspace przez parametr. Nie jest to pełna izolacja systemowa ani mechanizm autoryzacji użytkowników.

### Status API

```sh
curl http://127.0.0.1:8000/health/live
curl -i http://127.0.0.1:8000/health/ready
```

- `/health/live`: 200 i `{"status":"alive"}`. Nie odpytuje providera.
- `/health/ready`: 200, gdy runtime i fabryka agenta są gotowe, a provider zwraca dostępny wybrany model (lub jakikolwiek model przy automatycznym wyborze). Inaczej 503. Zwraca wyłącznie statusy komponentów, bez kluczy, ścieżek i treści wyjątków. Sprawdzenie modelu jest operacją sieciową ograniczoną skonfigurowanym timeoutem. Gotowość fabryki nie jest testem jakości odpowiedzi modelu.
- Przy wyłączonej Ollamie API nadal startuje i liveness działa, readiness ma 503, a błąd czatu jest zwracany jako zdarzenie strumienia.
- Endpointy audio w headless zwracają 503. Nie otwierają urządzeń ani nie pobierają modeli audio.

## Ryzyko, polityki i zgody narzędzi

Każde narzędzie ma w rejestrze zestaw etykiet ryzyka. Etykiety **nie wykluczają się**:

| Etykieta | Znaczenie |
|---|---|
| `read` | odczyt danych, bez zmian |
| `write` | tworzenie lub modyfikacja plików/stanu |
| `destructive` | usunięcie, nadpisanie albo uruchomienie kodu |
| `external` | dostęp do sieci lub zewnętrznego serwera/MCP |

Domyślne polityki (`allow`, `ask`, `deny`) są ustalane per etykieta, a następnie mogą być nadpisane per narzędzie:

```json
{
  "tool_risk_policies": {
    "read": "allow",
    "write": "ask",
    "destructive": "ask",
    "external": "ask"
  },
  "tool_policies": {
    "read_file": "allow",
    "delete_file": "deny",
    "mcp_echo_echo": "ask"
  }
}
```

Przy wielu etykietach wygrywa najbardziej restrykcyjna polityka (`deny > ask > allow`). Jawne `tool_policies` dla narzędzia ma pierwszeństwo. Polityki można zmieniać w GUI (Ustawienia → Narzędzia), przez `PUT /api/tools/policies` albo w `data/settings.json`.

Dla funkcji narzędzi bez `**kwargs` nieznane pola argumentów są pomijane (np. dodatkowe `region` w `search_web`), ale brak wymaganego argumentu nadal kończy się czytelnym `invalid_arguments`.

### Dozwolone katalogi

Narzędzia plikowe normalizują każdą ścieżkę (`~`, `..`, symlinki) i sprawdzają, czy po rozwiązaniu pozostaje w `NLP_WORKSPACE` lub jednym z katalogów `NLP_ALLOWED_DIRS`. Ścieżki spoza tych katalogów są odrzucane przed wykonaniem narzędzia. Ścieżki są normalizowane; jeśli katalog już istnieje, musi być katalogiem (może powstać później).

Dodatkowo narzędzia plikowe nie mogą odczytywać ani zmieniać `data/settings.json`, `data/mcp_servers.json`, `data/audit/**`, `data/logs/**`, `data/chats/**` ani `data/attachments/**`.

To nie jest pełna izolacja systemowa: `run_python` nadal wykonuje kod w podprocesie (z oczyszczonym środowiskiem z typowych sekretów) i nie zastępuje sandboxa OS.

### Zgody i pytania agenta

Gdy polityka narzędzia to `ask`, agent zatrzymuje wykonanie i wysyła zdarzenie `approval_required` z:

- `id` zgody,
- nazwą narzędzia i źródłem (`builtin`/MCP),
- etykietami ryzyka,
- bezpiecznymi argumentami (sekrety są maskowane, a treść plików/kodu nie jest pokazywana — tylko długość i liczba linii),
- czasem wygaśnięcia.

GUI pokazuje modal z wyborem „Odmów” / „Zezwól raz” / „Zezwól w sesji”. Dodatkowo w bąbelku pojawia się inline karta zgody, więc decyzję można podjąć nawet gdyby modal nie został zauważony. CLI robi to samo w terminalu. `allow_session` obowiązuje dla danego narzędzia do końca sesji czatu (reset czatu/modelu lub restart aplikacji).

Obok bloku „Proces myślenia” GUI pokazuje trwałe badge użytych narzędzi/MCP z etykietami ryzyka i statusem (oczekuje / wykonano / błąd / odmowa). Przy odmowie lub błędzie badge pokazuje też kod przyczyny (np. `invalid_arguments`, `user_denied`). Badge są zapisywane w historii czatu bez argumentów wywołania.

Po odmowie, timeoucie albo rozłączeniu klienta agent dostaje ustrukturyzowany wynik toola (`success=false`, `result.denied=true`, kod np. `user_denied`, `approval_timeout`, `client_disconnected`) i może kontynuować rozmowę bez prób obejścia odmowy.

Jeśli model po wyniku narzędzia zwróci tylko proces myślenia albo pustą treść, agent wykonuje jedną próbę ponowienia przez `ChatNoTools` z instrukcją odpowiedzi. Pusta wiadomość assistant nie trafia do kontekstu. Gdy ponowienie też nic nie zwróci, GUI pokazuje czytelny komunikat zamiast ciszy.

Jeśli model zapisze wywołanie narzędzia jako JSON w treści zamiast użyć `tool_calls`, agent loguje ostrzeżenie, czyści wersję roboczą odpowiedzi w GUI i wykonuje jedną korektę z prośbą o prawdziwy `tool_calls`. JSON nie jest automatycznie wykonywany. Jeśli korekta też się nie uda, GUI pokazuje błąd.

Diagnostyka w logach serwera pokazuje `Tool prepare` (polityka, wymagana zgoda, kod odmowy), `Approval required`, `Approval decision` oraz `Tool result` — bez argumentów wywołania i sekretów.

Agent może też użyć narzędzia `ask_user`, aby zadać pytanie z 2–6 opcjami. Domyślnie GUI i CLI pozwalają też dopisać własną odpowiedź (`allow_custom`); timeout/rozłączenie zwraca wynik `cancelled`.

### Logi i audit

Logi diagnostyczne trafiają do `data/logs/app.log` (`$CHAT_DATA_DIR/logs/app.log`) lub do ścieżki z `NLP_LOG_FILE`. Plik jest rotowany (5 MB × 3) i ma uprawnienia `0600`. Zawiera m.in. start runtime, `Context budget` (rozmiar kontekstu, rezerwa, tokeny narzędzi), `Tool prepare`, `Approval required`, `Approval decision`, `Tool result`, timeouty i informację o tekstowym tool callu — bez argumentów i sekretów. Samo „thinking” nie jest zapisywane do pliku.

Dla małych okien modelu (np. 4096 tokenów) schematy narzędzi i prompt narzędziowy są automatycznie kompaktowane: model dostaje skrócone opisy i strukturę parametrów, a pełne opisy pozostają w rejestrze i GUI. Można też uruchomić Ollamę z większym kontekstem, np. `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`.

Decyzje oraz operacje `destructive` są zapisywane osobno w `data/audit/tool_audit.jsonl` (JSONL, uprawnienia `0600`). Wpis zawiera m.in. czas UTC, narzędzie, ryzyko, decyzję, `chat_id` i `approval_id`. **Nie zawiera argumentów, treści plików, kodu ani wyników narzędzi.**

### Nowe endpointy API

- `GET /api/tools` — publiczne metadane narzędzi (ryzyko, polityka, źródło) i dozwolone katalogi;
- `PUT /api/tools/policies` — zapis `tool_policies` i `tool_risk_policies`;
- `POST /api/approvals/{id}` — `{"decision": "allow_once" | "allow_session" | "deny"}`;
- `POST /api/questions/{id}` — odpowiedź na `ask_user` (`answer`, opcjonalnie `option_index`, `cancelled`).

Kody błędów: `404` dla nieznanego ID, `409` dla już rozstrzygniętego, `410` dla wygasłego.

### Konfiguracja MCP

Serwer MCP może mieć `risk` (domyślnie `["external"]`) oraz `tools.ask`. Narzędzia z `tools.ask` są rejestrowane, ale wymagają zgody; `tools.deny` nadal blokuje rejestrację.

### Wstrzykiwanie zależności i testy

`ModelProvider` jest strukturalnym interfejsem metod konsumowanych przez `AgentFlow`. Obecny `OllamaHandler` realizuje ten kontrakt i udostępnia `close()`. `create_provider` buduje adapter z konfiguracją czasu i klucza. `AgentFlow`/`OllamaAgent` przyjmują `provider`; domyślna ścieżka dla dotychczasowych wywołań używa fabryki zamiast tworzenia konkretnego handlera wewnątrz flow. Runtime dostarcza fabrykę osobnych providerów dla sesji agenta, aby rozmowy nie nadpisywały sobie wyboru modelu. Każdy korzysta z tego samego rejestru narzędzi co jego flow.

```python
from core.api import create_app
from core.main import build_api_runtime
from core.RuntimeConfig import RuntimeConfig
from handlers.ToolHandle import ToolHandle

# fake_provider_factory(config, tools) zwraca obiekt zgodny z ModelProvider.
app = create_app(runtime_factory=lambda: build_api_runtime(
    RuntimeConfig(),
    provider_factory=fake_provider_factory,
    tool_factory=ToolHandle,
))
```

API zamyka runtime utworzony przez własny lifespan. Jeśli dostanie istniejący `runtime=...`, jego właściciel (np. desktopowy main) zamyka go sam. `AppRuntime.close()` jest idempotentne. Klienci providera są przechowywani do końca życia runtime i wtedy zamykani; długotrwałe serwery z bardzo dużą liczbą sesji mogą wymagać późniejszego usprawnienia usuwania nieaktywnych klientów.

```sh
python -m pip install pytest httpx
python -m pytest tests -q
```

Testy headless używają atrapy providera i rzeczywistej pętli AgentFlow oraz ASGI/TestClient. Nie wymagają działającej Ollamy ani bibliotek audio. Testują m.in. lifespan, zamykanie, dwie niezależne aplikacje, niedostępny model, ochronę sekretów, timeout adaptera, workspace i import bez opcjonalnych bibliotek GUI/audio.
