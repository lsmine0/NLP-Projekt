from handlers.ToolHandle import ToolHandle


def build_tools(*, files: bool = True, math: bool = True, web: bool = True) -> ToolHandle:
    """Zbuduj rejestr z wybranymi grupami narzędzi."""
    tool_handle = ToolHandle()
    if files:
        from tools.FileTools import register_tools

        register_tools(tool_handle)
    if math:
        from tools.MathTools import register_math_tools

        register_math_tools(tool_handle)
    if web:
        from tools.WebSearchTools import register_web_search_tools

        register_web_search_tools(tool_handle)
    return tool_handle
