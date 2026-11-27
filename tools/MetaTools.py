"""Meta-narzędzia: agent sam zagląda do swojego rejestru narzędzi.

Rejestr (``handlers.ToolHandle``) jest jedynym źródłem wiedzy o narzędziach.
Prompt systemowy przestał dublować katalog opisów: model widzi nazwy i krótkie
opisy w listie ``tools`` wysyłanej z żądaniem, a pełne szczegóły (parametry,
typy, wymagania) zwraca narzędzie ``list_tools``, gdy agent ich potrzebuje.

Dzięki temu stały kontekst jest krótszy o kilka tysięcy znaków, a prompt
narzędziowy mówi tylko, że taka introspekcja jest dostępna.
"""

from __future__ import annotations

from handlers.ToolHandle import ToolHandle

LIST_TOOLS_NAME = "list_tools"

LIST_TOOLS_DESCRIPTION = (
    "Lists the tools this agent can call. With no arguments it returns the names "
    "of all available tools; with tool_name it returns that tool's full "
    "description and parameter details (types, required fields). Call it before "
    "using an unfamiliar tool instead of guessing its arguments."
)


def register_meta_tools(tool_handle: ToolHandle, *, overwrite: bool = True) -> None:
    """Rejestruje ``list_tools`` - narzędzie introspekcji samego rejestru."""
    tool_handle.create_tool(
        name=LIST_TOOLS_NAME,
        description=LIST_TOOLS_DESCRIPTION,
        properties={
            "tool_name": {
                "type": "string",
                "description": (
                    "Optional name of one tool to inspect in full. Omit it to get "
                    "the overview list of all available tools."
                ),
            }
        },
        required=[],
        overwrite=overwrite,
        function=tool_handle.list_available_tools,
        risk=("read",),
        # Introspekcja rejestru nie zmienia stanu, nie dotyka sieci ani dysku.
        # Jawna polityka 'allow' trzyma ją dostępną nawet wtedy, gdy globalne
        # policy dla 'read' ustawiono na 'ask'; override z settings.json wciąż
        # ma pierwszeństwo (ToolHandle.get_policy).
        policy="allow",
        source="builtin",
    )
