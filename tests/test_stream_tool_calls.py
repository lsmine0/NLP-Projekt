"""Regression #51: strugane (fragmented) tool calls in the streaming path.

llama.cpp / llama-server wysyłają `delta.tool_calls` kawałkami z tym samym `index`:
id i nazwa tylko w pierwszym fragmente, `arguments` dosyłane piece-by-piece.
Stara pętla AgentFlow traktowała każdy fragment jako osobny tool call, więc
`json.loads` crashował na uciętym JSONe i cały turn ginął jako "Błąd modelu.".
Testy emulują dokładnie ten kształt payloadu (z `index` i bez niego).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.AgentAdapter import OllamaAgent
from handlers.ToolHandle import ToolHandle

CHAT = "c" * 32
_SENTINEL = object()


def delta_chunk(content=None, fragments=None):
    """Jeden chunk SSE: albo tekst, albo lista fragmentów tool_calls."""
    return NS(choices=[NS(delta=NS(content=content, reasoning_content=None,
                                  tool_calls=fragments))])


def fragment(call_id=None, name=None, arguments=None, index=_SENTINEL):
    """Kawałek tool_callu; `index` pominięty => server go nie podaje w ogóle."""
    function = NS(name=name if name is not None else "",
                  arguments=arguments if arguments is not None else "")
    piece = NS(id=call_id, function=function)
    if index is not _SENTINEL:
        piece.index = index
    return piece


def answer(text):
    """Skrypt iteracji, w której model już nie woła toola."""
    return [delta_chunk(content=text)]


class ScriptProvider:
    """Atrapa providera: kolejne ChatStream emitują kolejny pre-setowany skrypt."""

    def __init__(self, tools, scripts):
        self.ToolsHandle = tools
        self.base_url = "http://fake/v1"
        self.model = "scripted-model"
        self.scripts = list(scripts)
        self.payloads = []
        self.closed = 0

    def GetModels(self):
        return [NS(id=self.model)]

    def SetModel(self, name):
        self.model = name
        return True

    def GetNumCtx(self):
        return 8192

    def EnsureContextBudget(self, messages, **kwargs):
        return None

    def supports_vision(self):
        return True

    def ChatStream(self, messages, **kwargs):
        self.payloads.append(json.loads(json.dumps(messages)))
        script = self.scripts.pop(0) if self.scripts else answer("OK")
        yield from script

    def close(self):
        self.closed += 1


def echo_tools(capture):
    """Tool `probe` zapisuje argumenty, z którymi faktycznie został wywołany."""
    tools = ToolHandle()

    def probe(**kwargs):
        capture.append(dict(kwargs))
        return {"success": True, "echo": dict(kwargs)}

    tools.create_tool(name="probe", description="Echo tool for tests.",
                      properties={}, required=[], function=probe)
    return tools


def build_agent(tools, scripts):
    provider = ScriptProvider(tools, scripts)
    agent = OllamaAgent("http://fake/v1", "", system_prompt="SYS", tool_handle=tools,
                        provider=provider)
    return agent, provider


# -------------------------------------------------- fragmented arguments (index)
def test_fragmented_arguments_are_merged_into_one_call():
    capture = []
    scripts = [
        [delta_chunk(fragments=[fragment(call_id="call1", name="probe",
                                         arguments="{", index=0)]),
         delta_chunk(fragments=[fragment(arguments='"note": ', index=0)]),
         delta_chunk(fragments=[fragment(arguments='"ok"', index=0)]),
         delta_chunk(fragments=[fragment(arguments="}", index=0)])],
        answer("Tool zwrócił wynik."),
    ]
    agent, provider = build_agent(echo_tools(capture), scripts)

    events = list(agent.stream("Zwołaj probe"))

    assert capture == [{"note": "ok"}], "tool wywołany raz, ze scalonych argumentów"
    assert not [e for e in events if e.get("type") == "error"], "turn nie może ginąć"
    tool_messages = [m for m in agent.flow.context if m.get("role") == "tool"]
    assert len(tool_messages) == 1, "jeden call -> jedna message toola"


def test_merged_arguments_survive_the_context_round_trip():
    capture = []
    scripts = [
        [delta_chunk(fragments=[fragment(call_id="call1", name="probe",
                                         arguments="{", index=0)]),
         delta_chunk(fragments=[fragment(arguments='"note": "ok"', index=0)]),
         delta_chunk(fragments=[fragment(arguments="}", index=0)])],
        answer("Done."),
    ]
    agent, _ = build_agent(echo_tools(capture), scripts)
    list(agent.stream("Zwołaj probe"))

    assistant = [m for m in agent.flow.context
                 if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant) == 1
    arguments = assistant[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"note": "ok"}, "argumenty w kontekste muszą być JSONem"


# ------------------------------------------------- server without `index` field
def test_index_free_stream_still_merges_by_id_and_name():
    capture = []
    scripts = [
        [
            delta_chunk(fragments=[fragment(call_id="callA", name="probe", arguments="{")]),
            delta_chunk(fragments=[fragment(arguments='"note": "no-index"')]),
            delta_chunk(fragments=[fragment(arguments="}")]),
        ],
        answer("Done."),
    ]
    agent, _ = build_agent(echo_tools(capture), scripts)
    events = list(agent.stream("Zwołaj probe"))

    assert capture == [{"note": "no-index"}]
    assert not [e for e in events if e.get("type") == "error"]


# ------------------------------------------------------ parallel calls by index
def test_two_indexed_calls_execute_in_emitted_order():
    capture = []
    scripts = [
        [delta_chunk(fragments=[fragment(call_id="c0", name="probe",
                                         arguments='{"note": "first"}', index=0)]),
         delta_chunk(fragments=[fragment(call_id="c1", name="probe",
                                         arguments='{"note": "second"}', index=1)])],
        answer("Done."),
    ]
    agent, _ = build_agent(echo_tools(capture), scripts)
    list(agent.stream("Zwołaj probe dwa razy"))

    assert capture == [{"note": "first"}, {"note": "second"}]


# ------------------------------------------------- truncated JSON: soft failure
def test_truncated_arguments_do_not_kill_the_turn():
    capture = []
    scripts = [
        [delta_chunk(fragments=[fragment(call_id="call1", name="probe",
                                         arguments="{", index=0)]),
         delta_chunk(fragments=[fragment(arguments='"note":', index=0)])],
        answer("Zgłoszę, że argumenty były niepełne."),
    ]
    agent, _ = build_agent(echo_tools(capture), scripts)
    events = list(agent.stream("Zwołaj probe"))

    assert not [e for e in events if e.get("type") == "error"], "turn musi przeżyć"
    assert capture == [{}], "niepełny JSON -> puste argumenty, nie wyjątk"
