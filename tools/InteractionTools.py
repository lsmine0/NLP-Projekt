"""Interactive tool that lets the agent ask the user a question."""

from __future__ import annotations

from typing import Any

from handlers.ToolHandle import ToolHandle


def _ask_user_unavailable(**kwargs: Any) -> dict[str, Any]:
    """Fallback for direct registry calls; AgentFlow handles questions itself."""
    return {
        "success": False,
        "error": "Narzędzie ask_user jest dostępne tylko w interaktywnej sesji agenta.",
    }


def register_interaction_tools(tool_handle: ToolHandle) -> None:
    tool_handle.create_tool(
        name="ask_user",
        description=(
            "Asks the user a concise question when a decision or missing information "
            "blocks progress. Provide 2-6 short options; the user may always write a "
            "custom answer. Use this instead of guessing."
        ),
        properties={
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Between 2 and 6 concise answer options.",
            },
            "header": {
                "type": "string",
                "description": "Optional short heading for the question dialog.",
            },
            "allow_custom": {
                "type": "boolean",
                "description": "Whether the user may write a custom answer (default true).",
            },
        },
        required=["question", "options"],
        function=_ask_user_unavailable,
        risk=(),
        kind="user_question",
        source="builtin",
    )
