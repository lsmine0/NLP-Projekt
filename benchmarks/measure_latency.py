"""Diagnoza czasu odpowiedzi Ollamy: rozbicie TTFT na load / prompt eval / thinking / odpowiedź.

Uruchomienie (tylko biblioteka standardowa):
    python benchmarks/measure_latency.py --model qwen3:4b
    python benchmarks/measure_latency.py --model qwen3:4b --repeats 3 --out wyniki.json

Warianty "native"/"warm"/"cold" są mierzone natywnym /api/chat (zwraca load_duration,
prompt_eval_*, eval_*). Warianty "v1_*" idą przez endpoint OpenAI-compat (/v1), którego używa
OllamaHandler, żeby sprawdzić, które parametry sterujące thinkingiem są tam respektowane:
  v1_think_off       - "think": false (tak wysyłał OllamaHandler przed poprawką; /v1 to ignoruje),
  v1_reasoning_none  - "reasoning_effort": "none" (tak wysyła OllamaHandler po poprawce).
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import urllib.request

PROMPTS = {
    "B01": "Wyjaśnij w jednym zdaniu, czym jest NLP.",
    "B02": "Mam 3 pliki po 120 MB i 2 pliki po 70 MB. Ile miejsca zajmują razem? Podaj wynik w MB.",
    "T01": "Napisz krótkie powitanie po polsku.",
}

# (nazwa, endpoint, think, num_ctx, cold)
#   think=None -> parametr nie jest wysyłany (domyślne zachowanie aplikacji)
#   think="reasoning_none" -> (tylko v1) wysyłane "reasoning_effort": "none"
#   num_ctx=None -> domyślny kontekst serwera
VARIANTS = [
    ("cold_default", "native", None, None, True),
    ("warm_default", "native", None, None, False),
    ("warm_think_off", "native", False, None, False),
    ("warm_ctx16k", "native", None, 16384, False),
    ("v1_as_app", "v1", None, None, False),
    ("v1_think_off", "v1", False, None, False),
    ("v1_reasoning_none", "v1", "reasoning_none", None, False),
]


def post(url, body, stream=False, timeout=900):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def unload(base, model):
    with post(f"{base}/api/generate", {"model": model, "keep_alive": 0}) as r:
        r.read()
    time.sleep(1.0)


def warmup(base, model, num_ctx):
    body = {"model": model, "messages": [{"role": "user", "content": "ok"}],
            "stream": False, "think": False, "options": {"num_predict": 1}}
    if num_ctx:
        body["options"]["num_ctx"] = num_ctx
    with post(f"{base}/api/chat", body) as r:
        r.read()


def run_native(base, model, prompt, think, num_ctx):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    if think is not None:
        body["think"] = think
    if num_ctx:
        body["options"] = {"num_ctx": num_ctx}

    t0 = time.perf_counter()
    first_event = first_thinking = first_content = None
    thinking_chunks = content_chunks = 0
    final = {}
    with post(f"{base}/api/chat", body, stream=True) as r:
        for line in r:
            if not line.strip():
                continue
            msg = json.loads(line)
            now = time.perf_counter() - t0
            m = msg.get("message", {})
            if m.get("thinking"):
                thinking_chunks += 1
                first_thinking = first_thinking or now
                first_event = first_event or now
            if m.get("content"):
                content_chunks += 1
                first_content = first_content or now
                first_event = first_event or now
            if msg.get("done"):
                final = msg
    total = time.perf_counter() - t0
    ns = 1e9
    eval_s = final.get("eval_duration", 0) / ns
    eval_count = final.get("eval_count", 0)
    return {
        "wall_total_s": round(total, 2),
        "first_event_s": round(first_event, 2) if first_event else None,
        "ttft_s": round(first_content, 2) if first_content else None,
        "thinking_phase_s": round(first_content - first_thinking, 2)
        if first_content and first_thinking else 0.0,
        "thinking_chunks": thinking_chunks,
        "content_chunks": content_chunks,
        "load_s": round(final.get("load_duration", 0) / ns, 2),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_s": round(final.get("prompt_eval_duration", 0) / ns, 2),
        "eval_count": eval_count,
        "eval_s": round(eval_s, 2),
        "tok_per_s": round(eval_count / eval_s, 1) if eval_s else None,
        "done_reason": final.get("done_reason"),
    }


def run_v1(base, model, prompt, think):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    if think is not None:
        if think == "reasoning_none":
            body["reasoning_effort"] = "none"  # to, co OllamaHandler wysyła po poprawce
        else:
            body["think"] = think  # to, co OllamaHandler wysyłał przed poprawką
    t0 = time.perf_counter()
    first_event = first_thinking = first_content = None
    thinking_chunks = content_chunks = 0
    with post(f"{base}/v1/chat/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta", {})
            now = time.perf_counter() - t0
            if delta.get("reasoning") or delta.get("reasoning_content"):
                thinking_chunks += 1
                first_thinking = first_thinking or now
                first_event = first_event or now
            if delta.get("content"):
                content_chunks += 1
                first_content = first_content or now
                first_event = first_event or now
    total = time.perf_counter() - t0
    return {
        "wall_total_s": round(total, 2),
        "first_event_s": round(first_event, 2) if first_event else None,
        "ttft_s": round(first_content, 2) if first_content else None,
        "thinking_phase_s": round(first_content - first_thinking, 2)
        if first_content and first_thinking else 0.0,
        "thinking_chunks": thinking_chunks,
        "content_chunks": content_chunks,
    }


def server_state(base):
    state = {}
    try:
        state["version"] = get_json(f"{base}/api/version").get("version")
        state["ps"] = [
            {k: m.get(k) for k in ("name", "size", "size_vram", "context_length", "expires_at")}
            for m in get_json(f"{base}/api/ps").get("models", [])
        ]
    except Exception as e:  # noqa: BLE001
        state["error"] = str(e)
    state["env"] = {k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")}
    if platform.system() == "Darwin":
        for cmd in (["sysctl", "-n", "vm.swapusage"], ["memory_pressure", "-Q"]):
            try:
                state[cmd[0]] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:  # noqa: BLE001
                pass
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--variants", default=",".join(v[0] for v in VARIANTS))
    ap.add_argument("--out", default="latency_results.json")
    args = ap.parse_args()

    base = args.base.rstrip("/").removesuffix("/v1")
    prompt_ids = args.prompts.split(",")
    wanted = set(args.variants.split(","))
    results = {"model": args.model, "before": server_state(base), "runs": []}
    if "error" in results["before"]:
        raise SystemExit(
            f"Brak połączenia z Ollamą pod {base} ({results['before']['error']}).\n"
            "Uruchom skrypt na maszynie z Ollamą (np. MacBook M4: `ollama serve`) "
            "albo wskaż ją przez --base http://<IP>:11434 "
            "(na serwerze ustaw OLLAMA_HOST=0.0.0.0, żeby przyjmował połączenia z sieci)."
        )
    print(f"Ollama {results['before'].get('version')}  model={args.model}")

    for name, endpoint, think, num_ctx, cold in VARIANTS:
        if name not in wanted:
            continue
        for pid in prompt_ids:
            for rep in range(args.repeats):
                if cold:
                    unload(base, args.model)
                else:
                    warmup(base, args.model, num_ctx)
                try:
                    if endpoint == "native":
                        r = run_native(base, args.model, PROMPTS[pid], think, num_ctx)
                    else:
                        r = run_v1(base, args.model, PROMPTS[pid], think)
                except Exception as e:  # noqa: BLE001
                    r = {"error": str(e)}
                r.update(variant=name, prompt=pid, rep=rep)
                results["runs"].append(r)
                print(json.dumps(r, ensure_ascii=False))

    results["after"] = server_state(base)

    print("\n=== Mediany per wariant ===")
    cols = ["first_event_s", "ttft_s", "thinking_phase_s", "thinking_chunks", "content_chunks",
            "load_s", "prompt_eval_count", "prompt_eval_s", "eval_count", "tok_per_s", "wall_total_s"]
    print("variant".ljust(22) + "".join(c[:14].rjust(15) for c in cols))
    for name, *_ in VARIANTS:
        rows = [r for r in results["runs"] if r.get("variant") == name and "error" not in r]
        if not rows:
            continue
        line = name.ljust(22)
        for c in cols:
            vals = [r[c] for r in rows if isinstance(r.get(c), (int, float))]
            line += (f"{statistics.median(vals):.2f}" if vals else "-").rjust(15)
        print(line)
    print("\nps po teście:", json.dumps(results["after"].get("ps"), ensure_ascii=False))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Zapisano {args.out}")


if __name__ == "__main__":
    main()
