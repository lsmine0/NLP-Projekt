from __future__ import annotations

import pytest

from core.MermaidDiagrams import (
    DiagramSpecError,
    parse_diagram_spec,
    serialize_diagram,
)


def test_flowchart_serializer_generates_safe_ids_and_grouped_nodes() -> None:
    source = serialize_diagram(
        {
            "type": "flowchart",
            "direction": "LR",
            "groups": [{"id": "frontend", "label": "Frontend app"}],
            "nodes": [
                {"id": "browser", "label": "Web browser", "group": "frontend"},
                {"id": "api", "label": "Backend API"},
                {"id": "db", "label": "Database"},
            ],
            "edges": [
                {"from": "browser", "to": "api", "label": "HTTP request"},
                {"from": "api", "to": "db", "label": "Query"},
            ],
        }
    )

    assert source == (
        'flowchart LR\n'
        'subgraph g0["Frontend app"]\n'
        '    n0["Web browser"]\n'
        'end\n'
        'n1["Backend API"]\n'
        'n2["Database"]\n'
        'n0 -->|HTTP request| n1\n'
        'n1 -->|Query| n2'
    )


def test_flowchart_serializer_escapes_delimiters_and_rejects_bad_edges() -> None:
    source = serialize_diagram(
        {
            "type": "flowchart",
            "nodes": [
                {"id": "end", "label": 'Finish "now"'},
                {"id": "start", "label": "Start"},
            ],
            "edges": [{"from": "start", "to": "end", "label": "go|stop --> next"}],
        }
    )
    assert 'n0["Finish \'now\'"]' in source
    assert "n1 -->|go/stop → next| n0" in source

    with pytest.raises(DiagramSpecError, match="nieznany węzeł"):
        serialize_diagram(
            {
                "type": "flowchart",
                "nodes": [{"id": "start", "label": "Start"}],
                "edges": [{"from": "start", "to": "missing"}],
            }
        )

    with pytest.raises(DiagramSpecError, match="kierunek"):
        serialize_diagram(
            {
                "type": "flowchart",
                "direction": [],
                "nodes": [{"id": "start", "label": "Start"}],
            }
        )


def test_sequence_serializer_supports_response_and_alternatives() -> None:
    source = serialize_diagram(
        {
            "type": "sequence",
            "participants": [
                {"id": "browser", "label": "Przeglądarka"},
                {"id": "server", "label": "Serwer"},
            ],
            "messages": [
                {
                    "from": "browser",
                    "to": "server",
                    "label": "Wyślij dane",
                    "kind": "request",
                }
            ],
            "branches": [
                {
                    "label": "Dane poprawne",
                    "messages": [
                        {
                            "from": "server",
                            "to": "browser",
                            "label": "Sukces",
                            "kind": "response",
                        }
                    ],
                },
                {
                    "label": "Dane błędne",
                    "messages": [
                        {
                            "from": "server",
                            "to": "browser",
                            "label": "Błąd",
                            "kind": "response",
                        }
                    ],
                },
            ],
        }
    )

    assert source.startswith(
        "sequenceDiagram\n"
        "    participant p0 as Przeglądarka\n"
        "    participant p1 as Serwer\n"
        "    p0->>p1: Wyślij dane\n"
        "    alt Dane poprawne\n"
        "        p1-->>p0: Sukces\n"
        "    else Dane błędne\n"
        "        p1-->>p0: Błąd\n"
        "    end"
    )


def test_parse_diagram_spec_accepts_json_fence_and_rejects_prose() -> None:
    assert parse_diagram_spec('```json\n{"type":"flowchart"}\n```') == {
        "type": "flowchart"
    }

    with pytest.raises(DiagramSpecError, match="JSON-em"):
        parse_diagram_spec("The file is missing.")
