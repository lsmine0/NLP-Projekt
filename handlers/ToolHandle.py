import json
from typing import Any, Dict, List, Optional


class ToolHandle:
    """
    Klasa realizująca operacje CRUD na słowniku narzędzi (tools)
    zgodnie ze specyfikacją Ollama Tool Calling.
    """

    def __init__(self) -> None:
        # Słownik przechowujący narzędzia (klucz: nazwa narzędzia)
        self._tools: Dict[str, Dict[str, Any]] = {}

    # --- CREATE ---
    def create_tool(
        self,
        name: str,
        description: str,
        properties: Optional[Dict[str, Dict[str, Any]]] = None,
        required: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Tworzy nowe narzędzie i dodaje je do kolekcji."""
        if name in self._tools and not overwrite:
            raise ValueError(f"Narzędzie '{name}' już istnieje.")

        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "required": required or [],
                    "properties": properties or {},
                },
            },
        }

        self._tools[name] = tool_schema
        return tool_schema

    def add_parameter(
        self,
        tool_name: str,
        param_name: str,
        param_type: str,
        description: str,
        required: bool = True,
    ) -> Dict[str, Any]:
        """Dodaje pojedynczy parametr do istniejącego narzędzia."""
        if tool_name not in self._tools:
            raise KeyError(f"Narzędzie '{tool_name}' nie istnieje.")

        params = self._tools[tool_name]["function"]["parameters"]
        params["properties"][param_name] = {
            "type": param_type,
            "description": description,
        }

        if required and param_name not in params["required"]:
            params["required"].append(param_name)

        return self._tools[tool_name]

    # --- READ ---
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Pobiera pojedyncze narzędzie po nazwie."""
        return self._tools.get(name)

    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Zwraca listę narzędzi: [ {...}, {...} ]."""
        return list(self._tools.values())

    def get_tools_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Zwraca pełny słownik: {"tools": [ {...} ]}."""
        return {"tools": self.get_tools_list()}

    def to_json(self, indent: int = 2) -> str:
        """Zwraca słownik narzędzi sformatowany jako tekst JSON."""
        return json.dumps(self.get_tools_dict(), indent=indent, ensure_ascii=False)

    # --- UPDATE ---
    def update_tool(
        self,
        name: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Dict[str, Any]]] = None,
        required: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Aktualizuje wybrane pola istniejącego narzędzia."""
        if name not in self._tools:
            raise KeyError(f"Narzędzie '{name}' nie istnieje.")

        func_data = self._tools[name]["function"]

        if description is not None:
            func_data["description"] = description
        if properties is not None:
            func_data["parameters"]["properties"] = properties
        if required is not None:
            func_data["parameters"]["required"] = required

        return self._tools[name]

    def remove_parameter(self, tool_name: str, param_name: str) -> Dict[str, Any]:
        """Usuwa pojedynczy parametr z narzędzia."""
        if tool_name not in self._tools:
            raise KeyError(f"Narzędzie '{tool_name}' nie istnieje.")

        params = self._tools[tool_name]["function"]["parameters"]
        params["properties"].pop(param_name, None)
        if param_name in params["required"]:
            params["required"].remove(param_name)

        return self._tools[tool_name]

    # --- DELETE ---
    def delete_tool(self, name: str) -> bool:
        """Usuwa narzędzie. Zwraca True, jeśli narzędzie istniało i zostało usunięte."""
        return self._tools.pop(name, None) is not None

    def clear_tools(self) -> None:
        """Usuwa wszystkie narzędzia."""
        self._tools.clear()