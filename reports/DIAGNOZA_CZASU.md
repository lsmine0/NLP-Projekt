# Diagnoza czasu wykonywania benchmarku: hardware vs kod

Punkt wyjścia: `raport.json` (MacBook Air M4, 16 GB RAM, `qwen3:4b`): średni TTFT **111,03 s**, średni czas całkowity **117,97 s**, stały `first_event` ok. **7,8 s** w każdym teście, cały przebieg ok. 40 min.

Pomiary w tym dokumencie wykonano na **Windows 11, RTX 5070 Ti (16 GB VRAM), Ryzen 7 9800X3D, 32 GB RAM**, Ollama 0.40.15 (benchmark przed/po dla `qwen3:4b`) i 0.40.16 (pozostałe pomiary). Każdy wariant uruchomiono jednorazowo lub dwukrotnie; wyniki modelu różnią się między przebiegami. **Efekt poprawek na MacBooku nie został jeszcze zmierzony.**

## Wnioski

1. **Czas zjada thinking modelu, nie sprzęt ani komunikacja.** `qwen3:4b` w Ollamie to wersja z wymuszonym myśleniem: przed każdą odpowiedzią generuje średnio ok. 900 tokenów reasoningu (od ok. 300 do ok. 3000). Prompt, ładowanie modelu i sieć to ułamki sekundy.
2. **Thinkingu tego modelu nie da się wyłączyć parametrem.** Po wyłączeniu (`think: false` natywnie lub `reasoning_effort: "none"` przez `/v1`) model nadal rozważa tyle samo, tylko w polu odpowiedzi (`content`). Realnie przyspiesza dopiero model bez thinkingu (`qwen3:4b-instruct`).
3. **Kod dokładał kilka kosztów, które zostały poprawione** (sekcja „Przyczyny po stronie kodu”). Na Windowsie średni TTFT benchmarku spadł z **11,0 s do 4,8 s** na tym samym modelu.
4. **Różnica Mac ↔ Windows (111 s ↔ 11 s przed poprawkami) to głównie sprzęt:** RTX generuje ok. 205 tok/s, MacBook Air M4 szacunkowo kilka razy wolniej, a bez wentylatora dodatkowo zwalnia przy długim obciążeniu.

## Rozbicie czasu (`benchmarks/measure_latency.py`)

Mediana z 3 promptów (B01, B02, T01) × 2 powtórzenia, `qwen3:4b`, plik `windows_latency_qwen3-4b.json`. `thinking_ch` / `content_ch` to liczba fragmentów streamu (≈ tokenów) reasoningu i odpowiedzi.

| Wariant | TTFT | Pierwszy event | thinking_ch | content_ch | Load | tok/s |
|---|---|---|---|---|---|---|
| zimny start | 4,19 s | 1,61 s | 530 | 53 | 1,51 s | 206 |
| rozgrzany, domyślnie | 4,86 s | 0,04 s | 986 | 42 | 0 | 205 |
| rozgrzany, `num_ctx` 16384 | 4,22 s | 0,02 s | 863 | 45 | 0 | 204 |
| natywnie `think: false` | 0,02 s* | 0,02 s | 0 | 1085 | 0 | 204 |
| `/v1`, jak aplikacja | 3,71 s | 0,03 s | 782 | 39 | – | – |
| `/v1`, `think: false` | 4,25 s | 0,04 s | **866** | 44 | – | – |
| `/v1`, `reasoning_effort: "none"` | 0,04 s* | 0,04 s | 0 | 931 | – | – |

\* TTFT jest pozornie krótki: reasoning trafia do `content`, więc pierwszy „token odpowiedzi” to początek rozważań („Okay, the user wants…”). Czas całkowity się nie zmienia.

Odczyt:
- **Prompt eval i ładowanie modelu nie są problemem:** pierwszy event po rozgrzaniu to 0,02–0,04 s, zimny start ok. 1,5 s.
- **`num_ctx` nie wpływa na szybkość** przy tych promptach (4096 vs 16384 – różnica w granicach losowości).
- **`/v1` ignoruje `think: false`** (866 fragmentów reasoningu), a **respektuje `reasoning_effort: "none"`**.
- Dla porównania `qwen3:4b-instruct` (`windows_latency_qwen3-4b-instruct.json`): TTFT 0,03 s, ok. 50 tokenów na odpowiedź, zimny start 0,58 s.

## Przyczyny po stronie kodu

| # | Problem | Skutek | Status |
|---|---|---|---|
| 1 | `BenchmarkRunner` wstawiał do system promptu losową ścieżkę katalogu testowego, a w szablonie qwen3 definicje tooli stoją **za** system promptem | Ollama nie mogła użyć cache'u i przy każdym teście liczyła od zera ok. 1050 tokenów (system prompt + schematy tooli). Prawdopodobna przyczyna stałych ok. 7,8 s `first_event` na Macu | poprawione: stały system prompt, ścieżki względne |
| 2 | `OllamaHandler` wysyłał `think` przez `/v1`, gdzie jest po cichu ignorowany | sterowanie thinkingiem nie działało | poprawione: wysyłany jest też `reasoning_effort` |
| 3 | `GetNumCtx()` zwracał z `/api/show` **maksimum** modelu (262144 dla `qwen3:4b`) zamiast kontekstu, z którym serwer go uruchamia (domyślnie 4096) | `AgentFlow` liczył budżet dla 262144 tokenów: przycinanie historii nigdy się nie włączało, a Ollama po cichu obcinała prompt | poprawione: odczyt z `/api/ps`, a bez załadowanego modelu limit do `DEFAULT_NUM_CTX` |
| 4 | Historia czatu z GUI wracała do modelu razem z `<think>…</think>` | prompt rósł o cały dotychczasowy reasoning przy każdej wiadomości | poprawione w `AgentAdapter.load_history` |
| 5 | Brak warmupu | pierwszy request płacił za załadowanie modelu (1,5 s, przy zimnym dysku do ok. 40 s) | poprawione: `OllamaHandler.Warmup()`, wywoływany przez GUI i `BenchmarkRunner` |
| 6 | Adres `localhost` (tylko Windows) | Python najpierw próbuje IPv6, Ollama słucha na IPv4: **+2,03 s na każdy request** | poprawione: domyślnie `127.0.0.1`, `localhost` zamieniany automatycznie |

Sprawdzone i niewymagające zmian: model mieści się w całości w VRAM (3,2 GB przy kontekście 4096, 5,1 GB przy 16384), brak swapowania po stronie Windowsa.

## Wyniki benchmarku przed i po (`BenchmarkRunner.py`, 23 testy, 19 wykonanych)

| | Przed poprawkami | Po poprawkach | Po poprawkach, `qwen3:4b-instruct` |
|---|---|---|---|
| Plik | `windows_before_qwen3-4b.json` | `windows_after_minimal_qwen3-4b.json` | `windows_after_minimal_qwen3-4b-instruct.json` |
| Średni TTFT | 10,99 s | **4,83 s** | **0,12 s** |
| Średni czas całkowity | 11,82 s | 5,26 s | 0,49 s |
| `failed` | 4 (F01, E02, SEC02, SEC11) | 1 (E02) | 2 (B03, E01) |
| `needs_review` | 15 | 18 | 17 |
| `blocked` | 4 | 4 | 4 |

Uwagi:
- Wersja „przed” to niezmieniony kod z `main` uruchomiony na tej samej maszynie. Część zysku (ok. 2 s na request) to poprawka `localhost`, która dotyczy tylko Windowsa.
- Spadek liczby `failed` z 4 do 1 nie jest dowodem poprawy jakości – przy jednym przebiegu mieści się w losowości modelu.
- `qwen3:4b-instruct` jest ok. 40× szybszy, ale **oblał testy zaliczane przez wersję z thinkingiem**: B03 (zwrócił samą tablicę zamiast obiektu z kluczem `zastosowania`) i E01 (nie wywołał `read_file` dla nieistniejącego pliku).
- `blocked` (STT01, TTS01, SEC06, SEC08) to celowe zachowanie runnera: brak integracji audio i niewypełnione placeholdery w scenariuszach.

## Wymagana konfiguracja serwera Ollamy

Przez `/v1` nie da się ustawić kontekstu ani czasu trzymania modelu w zapytaniu – robi się to po stronie serwera. Ollama domyślnie uruchamia model z kontekstem **4096**, a to za mało dla agenta: `AgentFlow` rezerwuje min. 8000 tokenów na odpowiedź, prompt z toolami ma ok. 1100 tokenów, a sam reasoning dochodzi do ok. 3000.

macOS (potem restart aplikacji Ollama):

```bash
launchctl setenv OLLAMA_CONTEXT_LENGTH 16384
launchctl setenv OLLAMA_KEEP_ALIVE 30m
```

Windows (potem restart Ollamy z zasobnika):

```bash
setx OLLAMA_CONTEXT_LENGTH 16384
setx OLLAMA_KEEP_ALIVE 30m
```

Koszt: model zajmuje ok. 5,1 GB zamiast 3,2 GB – na MacBooku z 16 GB RAM mieści się bezpiecznie.

## Rekomendacje

1. **Ustawić `OLLAMA_CONTEXT_LENGTH=16384` i `OLLAMA_KEEP_ALIVE=30m`** na maszynach z Ollamą.
2. **Decyzja zespołu: model domyślny.** `qwen3:4b-instruct` daje odpowiedź w ułamku sekundy (szacunkowo kilka sekund na M4), ale słabiej trzyma się instrukcji formatu i narzędzi. Możliwy kompromis: instruct dla prostych zapytań, wersja z thinkingiem dla zadań wymagających rozumowania; przy B03 strukturę JSON można wymusić parametrem `format` natywnego API.
3. **Powtórzyć benchmark na MacBooku** po poprawkach i ustawieniu serwera, porównać z `raport.json`:
   ```bash
   python BenchmarkRunner.py --model qwen3:4b --output reports/mac_after_qwen3-4b.json
   python BenchmarkRunner.py --model qwen3:4b-instruct --output reports/mac_after_qwen3-4b-instruct.json
   python benchmarks/measure_latency.py --model qwen3:4b --out reports/mac_latency_qwen3-4b.json
   ```
   Podczas przebiegu sprawdzić `ollama ps` (kolumna `PROCESSOR` powinna pokazywać `100% GPU`) oraz Monitor aktywności → Pamięć (presja pamięci, swap).
4. **Do rozważenia jako osobne zadanie:** przejście `OllamaHandler` na natywne `/api/chat` dla Ollamy (z zachowaniem `/v1` dla innych serwerów). Dałoby `num_ctx`/`keep_alive` per zapytanie i statystyki czasu w odpowiedzi, ale to zmiana architektury handlera, wykraczająca poza to zadanie.

## Status checklisty z issue

| Punkt | Status |
|---|---|
| Rozdzielić czasy: zimny start, prompt eval, thinking, odpowiedź, rundy tooli | ✅ `measure_latency.py` (bez rund tooli) + `BenchmarkRunner` (z rundami tooli) |
| Porównać `first_event_seconds` i `ttft_seconds` | ✅ różnica to faza thinkingu |
| Wpływ `think` on/off | ✅ dla tego modelu tylko przenosi reasoning do odpowiedzi |
| Wpływ `num_ctx` | ✅ brak wpływu na szybkość; wpływ na poprawność budżetu kontekstu (przyczyna #3) |
| Wpływ `keep_alive` | ✅ zimny vs rozgrzany start: ok. 1,5 s |
| Wpływ temperatury | ❌ nie badano |
| Rozmiar system promptu / liczba tooli | ⚠️ częściowo: prompt z toolami ok. 1050 tokenów, po poprawce #1 liczony z cache w 0,02 s; liczby tooli nie zmieniano |
| Czy `OllamaHandler` wysyła wszystkie toole i długi prompt | ⚠️ tak, wysyła wszystkie zarejestrowane toole w każdym zapytaniu; po poprawce #1 koszt jest pomijalny, więc tego nie zmieniano |
| RAM i swap przy 16 GB | ⚠️ sprawdzone tylko na Windowsie (model w całości w VRAM); na Macu do sprawdzenia przy kolejnym przebiegu |
| Równoległe sesje / inne procesy | ❌ nie badano |
| Poprawki niskiego ryzyka: warmup, `keep_alive`, thinking, `num_ctx`, krótszy prompt | ✅ warmup, `reasoning_effort`, poprawka `GetNumCtx`, stały system prompt; `num_ctx` i `keep_alive` przez konfigurację serwera |
| Rerun benchmarku i porównanie przed/po | ⚠️ zrobione na Windowsie; na MacBooku do wykonania |
