from handlers.ToolHandle import ToolHandle


def build_tools(
    *,
    files: bool = True,
    math: bool = True,
    web: bool = True,
    code: bool = True,
    workspace_dir=None,
) -> ToolHandle:
    """Zbuduj rejestr z wybranymi grupami narzędzi."""
    tool_handle = ToolHandle()
    if files:
        from tools.FileTools import register_tools

        register_tools(tool_handle)
        if workspace_dir is not None:
            from pathlib import Path
            from functools import wraps
            import inspect
            from tools.FileTools import AVAILABLE_FILE_TOOLS
            root = Path(workspace_dir).resolve()
            def confined(function):
                @wraps(function)
                def call(**kwargs):
                    bound = inspect.signature(function).bind(**kwargs)
                    for key in ('path', 'source', 'destination'):
                        if key in bound.arguments:
                            path = Path(bound.arguments[key])
                            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
                            if not path.is_relative_to(root):
                                raise PermissionError('Ścieżka poza workspace.')
                            bound.arguments[key] = str(path)
                    return function(**bound.arguments)
                return call
            for name, function in AVAILABLE_FILE_TOOLS.items():
                tool_handle.register_function(name, confined(function))
    if math:
        from tools.MathTools import register_math_tools

        register_math_tools(tool_handle)
    if web:
        from tools.WebSearchTools import register_web_search_tools

        register_web_search_tools(tool_handle)
    if code:
        from tools.CodeTools import register_code_tools

        register_code_tools(tool_handle, workspace_dir=workspace_dir)
    return tool_handle
