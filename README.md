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
| `NLP_WORKSPACE` | Katalog dla narzędzi plikowych i kodowania, domyślnie bieżący katalog |
| `CHAT_DATA_DIR` | Katalog ustawień, czatów i załączników, domyślnie `data/` w repozytorium |

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
python -m pytest tests/test_headless_api.py tests/test_main.py tests/test_webgui_attachments.py tests/testAudioAdapters.py tests/test_prompt_builder.py tests/test_benchmark_runner.py -q
```

Testy headless używają atrapy providera i rzeczywistej pętli AgentFlow oraz ASGI/TestClient. Nie wymagają działającej Ollamy ani bibliotek audio. Testują m.in. lifespan, zamykanie, dwie niezależne aplikacje, niedostępny model, ochronę sekretów, timeout adaptera, workspace i import bez opcjonalnych bibliotek GUI/audio.
