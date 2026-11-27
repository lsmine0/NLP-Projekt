from pathlib import Path
from typing import Iterable

from handlers.ToolHandle import ToolHandle


def build_tools(
    *,
    files: bool = True,
    math: bool = True,
    web: bool = True,
    code: bool = True,
    images: bool = True,
    interactions: bool = True,
    meta: bool = True,
    workspace_dir=None,
    allowed_dirs: Iterable[str | Path] | None = None,
    policy_overrides: dict[str, str] | None = None,
    risk_policies: dict[str, str] | None = None,
    protected_paths: Iterable[str | Path] | None = None,
) -> ToolHandle:
    """Zbuduj rejestr z wybranymi grupami narzędzi.

    ``workspace_dir`` is the primary directory used for relative paths. Every
    file tool is validated against ``workspace_dir`` plus optional
    ``allowed_dirs``; paths escaping those roots are denied before execution.

    ``meta`` registers ``list_tools``, the introspection tool that lets the agent
    discover its own registry (names, short descriptions and - on request - full
    parameter details). It is infrastructure rather than a feature group, so it
    stays on by default even when every other group is disabled.
    """

    roots: list[str | Path] = []
    if workspace_dir is not None:
        roots.append(workspace_dir)
    if allowed_dirs is not None:
        roots.extend(allowed_dirs)
    # Bez jawnego workspace/allowed_dirs biblioteka zachowuje dotychczasowe
    # działanie (brak konfinacji). Aplikacja zawsze przekazuje workspace.
    allowed_directories = roots or None
    primary_root = workspace_dir or (roots[0] if roots else None)

    tool_handle = ToolHandle(
        allowed_directories=allowed_directories,
        primary_root=primary_root,
        policy_overrides=policy_overrides,
        risk_policies=risk_policies,
        protected_paths=protected_paths,
    )

    # Rejestr narzędzi = single source of truth; list_tools musi być zarejestrowany
    # przed pozosta grupami, żeby był pierwszy w schemie i w logach.
    if meta:
        from tools.MetaTools import register_meta_tools

        register_meta_tools(tool_handle)
    if files:
        from tools.FileTools import register_tools

        register_tools(tool_handle)
    if images:
        # #51: inspekcja załączników po ID (zakres sesji - handlers.ToolHandle).
        from tools.ImageTools import register_tools as register_image_tools

        register_image_tools(tool_handle)
    if math:
        from tools.MathTools import register_math_tools

        register_math_tools(tool_handle)
    if web:
        from tools.WebSearchTools import register_web_search_tools

        register_web_search_tools(tool_handle)
    if code:
        from tools.CodeTools import register_code_tools

        register_code_tools(
            tool_handle,
            workspace_dir=workspace_dir or (roots[0] if roots else None),
        )
    if interactions:
        from tools.InteractionTools import register_interaction_tools

        register_interaction_tools(tool_handle)
    return tool_handle
