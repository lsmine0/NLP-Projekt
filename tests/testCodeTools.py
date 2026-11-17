from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handlers.ToolHandle import ToolHandle
from tools.CodeTools import (
    ISOLATION_WARNING,
    configure_code_tools,
    edit_file,
    register_code_tools,
    run_python,
)
from tools.FileTools import (
    create_file,
    list_directory,
    read_file,
    search_in_files,
    write_file,
)
from tools.ToolBuilder import build_tools


class CodeToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        configure_code_tools(workspace_dir=None, allow_execution=False, approval_callback=None)
        self.addCleanup(
            lambda: configure_code_tools(
                workspace_dir=None,
                allow_execution=False,
                approval_callback=None,
            )
        )

    def test_edit_file_replaces_unique_fragment(self) -> None:
        file_path = self.workspace / "calc.py"
        file_path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        result = edit_file(
            "calc.py",
            "return a - b",
            "return a + b",
            workspace_dir=self.workspace,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["replacements"], 1)
        self.assertEqual(
            file_path.read_text(encoding="utf-8"),
            "def add(a, b):\n    return a + b\n",
        )

    def test_edit_file_rejects_missing_empty_and_ambiguous_target(self) -> None:
        file_path = self.workspace / "notes.txt"
        original = "line alpha\nline shared\nline beta\nline shared\n"
        file_path.write_text(original, encoding="utf-8")

        missing = edit_file(
            "notes.txt",
            "nonexistent",
            "replacement",
            workspace_dir=self.workspace,
        )
        self.assertFalse(missing["success"])
        self.assertIn("not found", missing["error"])

        empty = edit_file(
            "notes.txt",
            "",
            "replacement",
            workspace_dir=self.workspace,
        )
        self.assertFalse(empty["success"])
        self.assertIn("empty", empty["error"])

        ambiguous = edit_file(
            "notes.txt",
            "line shared",
            "line updated",
            workspace_dir=self.workspace,
        )
        self.assertFalse(ambiguous["success"])
        self.assertIn("2 times", ambiguous["error"])
        self.assertEqual(file_path.read_text(encoding="utf-8"), original)

    def test_edit_file_validates_python_syntax_without_overwriting(self) -> None:
        file_path = self.workspace / "broken.py"
        original = "def greet():\n    return 'hello'\n"
        file_path.write_text(original, encoding="utf-8")

        result = edit_file(
            "broken.py",
            "    return 'hello'",
            "    return ('hello'",
            workspace_dir=self.workspace,
        )

        self.assertFalse(result["success"])
        self.assertIn("SyntaxError", result["error"])
        self.assertEqual(file_path.read_text(encoding="utf-8"), original)

    def test_edit_file_normalizes_and_preserves_crlf_and_lf(self) -> None:
        crlf_file = self.workspace / "crlf.py"
        crlf_file.write_bytes(b"def f():\r\n    x = 1\r\n    return x\r\n")

        res_crlf = edit_file(
            "crlf.py",
            "    x = 1\n    return x",
            "    x = 42\n    return x",
            workspace_dir=self.workspace,
        )
        self.assertTrue(res_crlf["success"])
        self.assertEqual(
            crlf_file.read_bytes(),
            b"def f():\r\n    x = 42\r\n    return x\r\n",
        )

        lf_file = self.workspace / "lf.py"
        lf_file.write_bytes(b"def g():\n    y = 1\n    return y\n")

        res_lf = edit_file(
            "lf.py",
            "    y = 1\r\n    return y",
            "    y = 99\r\n    return y",
            workspace_dir=self.workspace,
        )
        self.assertTrue(res_lf["success"])
        self.assertEqual(
            lf_file.read_bytes(),
            b"def g():\n    y = 99\n    return y\n",
        )

    def test_edit_file_blocks_traversal_and_outside_paths(self) -> None:
        secret = self.outside / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")

        traversal = edit_file(
            "../outside/secret.txt",
            "top secret",
            "hacked",
            workspace_dir=self.workspace,
        )
        self.assertFalse(traversal["success"])
        self.assertIn("outside", traversal["error"])
        self.assertEqual(secret.read_text(encoding="utf-8"), "top secret")

        absolute = edit_file(
            str(secret),
            "top secret",
            "hacked",
            workspace_dir=self.workspace,
        )
        self.assertFalse(absolute["success"])
        self.assertIn("outside", absolute["error"])
        self.assertEqual(secret.read_text(encoding="utf-8"), "top secret")

    def test_edit_file_blocks_symlinks(self) -> None:
        target_inside = self.workspace / "real.txt"
        target_inside.write_text("hello world", encoding="utf-8")
        link_inside = self.workspace / "link_inside.txt"

        target_outside = self.outside / "outside.txt"
        target_outside.write_text("outside data", encoding="utf-8")
        link_outside = self.workspace / "link_outside.txt"

        try:
            link_inside.symlink_to(target_inside)
            link_outside.symlink_to(target_outside)
        except OSError:
            # Windows bez Developer Mode / uprawnień administratora może blokować tworzenie symlinków.
            with patch.object(Path, "is_symlink", autospec=True) as mock_is_symlink:
                mock_is_symlink.side_effect = lambda p: p.name == "real.txt"
                mocked_result = edit_file(
                    "real.txt",
                    "hello",
                    "hi",
                    workspace_dir=self.workspace,
                )
            self.assertFalse(mocked_result["success"])
            self.assertIn("Symlinks are not allowed", mocked_result["error"])
            return

        res_in = edit_file("link_inside.txt", "hello", "hi", workspace_dir=self.workspace)
        self.assertFalse(res_in["success"])
        self.assertIn("Symlinks are not allowed", res_in["error"])
        self.assertEqual(target_inside.read_text(encoding="utf-8"), "hello world")

        res_out = edit_file("link_outside.txt", "outside", "changed", workspace_dir=self.workspace)
        self.assertFalse(res_out["success"])
        self.assertEqual(target_outside.read_text(encoding="utf-8"), "outside data")

    def test_run_python_requires_user_approval_and_logs_warning(self) -> None:
        with patch("tools.CodeTools.Logger.printWarning") as mock_warn:
            denied_default = run_python("print('ok')", workspace_dir=self.workspace)
        self.assertFalse(denied_default["success"])
        self.assertIsNone(denied_default["exit_code"])
        self.assertIn("approval", denied_default["error"].lower())
        self.assertEqual(denied_default["isolation_warning"], ISOLATION_WARNING)
        mock_warn.assert_called_once_with(ISOLATION_WARNING)

        denied_cb = run_python(
            "print('ok')",
            workspace_dir=self.workspace,
            approval_callback=lambda code: False,
        )
        self.assertFalse(denied_cb["success"])
        self.assertIn("rejected", denied_cb["error"].lower())

        seen_codes: list[str] = []

        def approve(code: str) -> bool:
            seen_codes.append(code)
            return True

        approved = run_python(
            "print('hello sandbox')",
            workspace_dir=self.workspace,
            approval_callback=approve,
        )
        self.assertTrue(approved["success"])
        self.assertEqual(approved["exit_code"], 0)
        self.assertIn("hello sandbox", approved["stdout"])
        self.assertEqual(seen_codes, ["print('hello sandbox')"])
        self.assertEqual(approved["isolation_warning"], ISOLATION_WARNING)

    def test_run_python_executes_in_workspace_handles_errors_and_truncates_output(self) -> None:
        (self.workspace / "helper_mod.py").write_text("VALUE = 123\n", encoding="utf-8")

        ok = run_python(
            "import helper_mod; print(helper_mod.VALUE)",
            workspace_dir=self.workspace,
            allow_execution=True,
        )
        self.assertTrue(ok["success"])
        self.assertEqual(ok["exit_code"], 0)
        self.assertEqual(ok["stdout"].strip(), "123")

        failing = run_python(
            "import sys; sys.stderr.write('boom\\n'); sys.exit(3)",
            workspace_dir=self.workspace,
            allow_execution=True,
        )
        self.assertFalse(failing["success"])
        self.assertEqual(failing["exit_code"], 3)
        self.assertIn("boom", failing["stderr"])

        noisy = run_python(
            "print('A' * 25000)",
            workspace_dir=self.workspace,
            allow_execution=True,
        )
        self.assertTrue(noisy["success"])
        self.assertTrue(noisy["truncated"])
        self.assertLess(len(noisy["stdout"]), 11_000)
        self.assertIn("Output truncated", noisy["stdout"])

    def test_run_python_timeout(self) -> None:
        result = run_python(
            "import time; time.sleep(5)",
            timeout=0.3,
            workspace_dir=self.workspace,
            allow_execution=True,
        )
        self.assertFalse(result["success"])
        self.assertIsNone(result["exit_code"])
        self.assertIn("timed out", result["error"].lower())

    def test_compatibility_with_file_tools(self) -> None:
        file_path = self.workspace / "module.py"
        created = create_file(str(file_path), "def mult(x, y):\n    return x + y\n")
        self.assertTrue(created["success"])

        edited = edit_file(
            "module.py",
            "return x + y",
            "return x * y",
            workspace_dir=self.workspace,
        )
        self.assertTrue(edited["success"])

        read_res = read_file(edited["path"])
        self.assertTrue(read_res["success"])
        self.assertIn("return x * y", read_res["content"])

        list_res = list_directory(str(self.workspace))
        self.assertTrue(list_res["success"])
        self.assertEqual([e["name"] for e in list_res["entries"]], ["module.py"])

        search_res = search_in_files(str(self.workspace), "x * y")
        self.assertTrue(search_res["success"])
        self.assertEqual(len(search_res["matches"]), 1)
        self.assertEqual(search_res["matches"][0]["line_number"], 2)

    def test_tool_handle_registration_and_parameter_isolation(self) -> None:
        (self.workspace / "app.py").write_text("X = 1\n", encoding="utf-8")
        handle = ToolHandle()
        register_code_tools(
            handle,
            workspace_dir=self.workspace,
            approval_callback=lambda code: True,
        )

        names = [item["function"]["name"] for item in handle.get_tools_list()]
        self.assertEqual(names, ["edit_file", "run_python"])
        self.assertNotIn("run_command", names)

        edit_tool = handle.get_tool("edit_file")
        self.assertIsNotNone(edit_tool)
        assert edit_tool is not None
        edit_props = edit_tool["function"]["parameters"]["properties"]
        self.assertNotIn("workspace_dir", edit_props)

        run_tool = handle.get_tool("run_python")
        self.assertIsNotNone(run_tool)
        assert run_tool is not None
        run_props = run_tool["function"]["parameters"]["properties"]
        self.assertNotIn("allow_execution", run_props)
        self.assertNotIn("workspace_dir", run_props)

        edit_out = json.loads(
            handle.execute_tool(
                "edit_file",
                {"path": "app.py", "target": "X = 1", "replacement": "X = 2"},
            )
        )
        self.assertTrue(edit_out["success"])

        run_out = json.loads(
            handle.execute_tool(
                "run_python",
                {"code": "import app; print(app.X)"},
            )
        )
        self.assertTrue(run_out["success"])
        self.assertEqual(run_out["result"]["stdout"].strip(), "2")

        bypass_attempt = json.loads(
            handle.execute_tool(
                "run_python",
                {"code": "print(1)", "allow_execution": True},
            )
        )
        self.assertFalse(bypass_attempt["success"])

    def test_tool_builder_includes_code_tools_by_default(self) -> None:
        with_code = build_tools(files=False, math=False, web=False, code=True)
        self.assertEqual(
            [t["function"]["name"] for t in with_code.get_tools_list()],
            ["edit_file", "run_python"],
        )
        without_code = build_tools(files=False, math=False, web=False, code=False)
        self.assertEqual(without_code.get_tools_list(), [])

    def test_edit_file_validates_json_syntax_and_edits_other_languages(self) -> None:
        json_file = self.workspace / "config.json"
        original_json = '{\n  "enabled": false\n}\n'
        json_file.write_text(original_json, encoding="utf-8")

        bad = edit_file(
            "config.json",
            '"enabled": false',
            '"enabled": tru',
            workspace_dir=self.workspace,
        )
        self.assertFalse(bad["success"])
        self.assertIn("SyntaxError", bad["error"])
        self.assertEqual(json_file.read_text(encoding="utf-8"), original_json)

        good = edit_file(
            "config.json",
            '"enabled": false',
            '"enabled": true',
            workspace_dir=self.workspace,
        )
        self.assertTrue(good["success"])
        self.assertEqual(json.loads(json_file.read_text(encoding="utf-8")), {"enabled": True})

        js_file = self.workspace / "app.js"
        js_file.write_text("const x = 1;\nconsole.log(x);\n", encoding="utf-8")
        js_res = edit_file("app.js", "const x = 1;", "const x = 42;", workspace_dir=self.workspace)
        self.assertTrue(js_res["success"])
        self.assertIn("const x = 42;", js_file.read_text(encoding="utf-8"))

    def test_run_python_supports_allowed_languages_and_rejects_shells(self) -> None:
        for shell in ("bash", "powershell", "cmd"):
            rejected = run_python(
                "echo hi",
                workspace_dir=self.workspace,
                language=shell,
                allow_execution=True,
            )
            self.assertFalse(rejected["success"])
            self.assertIn("Unsupported language", rejected["error"])

        with patch("tools.CodeTools.shutil.which", return_value=None):
            missing = run_python(
                "console.log(1)",
                workspace_dir=self.workspace,
                language="javascript",
                allow_execution=True,
            )
        self.assertFalse(missing["success"])
        self.assertIn("was not found in PATH", missing["error"])

        fake_completed = subprocess.CompletedProcess(
            args=["node", "-e", "console.log(42)"],
            returncode=0,
            stdout="42\n",
            stderr="",
        )
        with (
            patch("tools.CodeTools.shutil.which", return_value="C:/Program Files/nodejs/node.exe"),
            patch("tools.CodeTools.subprocess.run", return_value=fake_completed) as mock_run,
        ):
            ok = run_python(
                "console.log(42)",
                workspace_dir=self.workspace,
                language="javascript",
                allow_execution=True,
            )
        self.assertTrue(ok["success"])
        self.assertEqual(ok["stdout"].strip(), "42")
        self.assertEqual(ok["exit_code"], 0)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0], ["node", "-e", "console.log(42)"])

    def test_run_python_supports_cpp_compilation_and_execution(self) -> None:
        cpp_code = '#include <iostream>\nint main() { std::cout << "hello cpp\\n"; return 0; }\n'

        with patch("tools.CodeTools.shutil.which", return_value=None):
            missing = run_python(
                cpp_code,
                workspace_dir=self.workspace,
                language="cpp",
                allow_execution=True,
            )
        self.assertFalse(missing["success"])
        self.assertIn("g++", missing["error"])

        compile_fail = subprocess.CompletedProcess(
            args=["g++"],
            returncode=1,
            stdout="",
            stderr="snippet.cpp:1:1: error: expected ';'\n",
        )
        with (
            patch("tools.CodeTools.shutil.which", return_value="C:/msys64/mingw64/bin/g++.exe"),
            patch("tools.CodeTools.subprocess.run", return_value=compile_fail) as mock_fail,
        ):
            failed = run_python(
                "broken cpp",
                workspace_dir=self.workspace,
                language="cpp",
                allow_execution=True,
            )
        self.assertFalse(failed["success"])
        self.assertTrue(failed["error"].startswith("Compilation failed:"))
        self.assertIn("expected ';'", failed["stderr"])
        self.assertEqual(mock_fail.call_count, 1)

        compile_ok = subprocess.CompletedProcess(args=["g++"], returncode=0, stdout="", stderr="")
        run_ok = subprocess.CompletedProcess(args=["snippet"], returncode=0, stdout="hello cpp\n", stderr="")
        with (
            patch("tools.CodeTools.shutil.which", return_value="C:/msys64/mingw64/bin/g++.exe"),
            patch("tools.CodeTools.subprocess.run", side_effect=[compile_ok, run_ok]) as mock_ok,
        ):
            ok = run_python(
                cpp_code,
                workspace_dir=self.workspace,
                language="cpp",
                allow_execution=True,
            )
        self.assertTrue(ok["success"])
        self.assertEqual(ok["stdout"], "hello cpp\n")
        self.assertEqual(ok["exit_code"], 0)
        self.assertEqual(mock_ok.call_count, 2)
        self.assertEqual(mock_ok.call_args_list[0].args[0][0], "g++")

    def test_edit_file_validates_cpp_syntax_with_fallback(self) -> None:
        cpp_file = self.workspace / "main.cpp"
        original = "int main() { return 0; }\n"
        cpp_file.write_text(original, encoding="utf-8")

        # Brak kompilatora g++ w systemie -> edycja zapisuje się normalnie bez błędu.
        with patch("tools.CodeTools.shutil.which", return_value=None):
            no_compiler = edit_file(
                "main.cpp",
                "return 0;",
                "return 1;",
                workspace_dir=self.workspace,
            )
        self.assertTrue(no_compiler["success"])
        self.assertIn("return 1;", cpp_file.read_text(encoding="utf-8"))

        # Gdy g++ jest dostępny i wykryje błąd składni -> plik nie zostaje nadpisany.
        syntax_fail = subprocess.CompletedProcess(
            args=["g++"],
            returncode=1,
            stdout="",
            stderr="<stdin>:1:24: error: expected ';' before '}' token\n",
        )
        with (
            patch("tools.CodeTools.shutil.which", return_value="C:/msys64/mingw64/bin/g++.exe"),
            patch("tools.CodeTools.subprocess.run", return_value=syntax_fail) as mock_run,
        ):
            bad_cpp = edit_file(
                "main.cpp",
                "return 1;",
                "return 1",
                workspace_dir=self.workspace,
            )
        self.assertFalse(bad_cpp["success"])
        self.assertIn("SyntaxError", bad_cpp["error"])
        self.assertIn("expected ';'", bad_cpp["error"])
        self.assertIn("return 1;", cpp_file.read_text(encoding="utf-8"))
        self.assertEqual(
            mock_run.call_args.args[0],
            ["g++", "-fsyntax-only", "-x", "c++", "-"],
        )

    def test_edit_file_creates_new_code_file_with_validation_and_write_file(self) -> None:
        new_py = self.workspace / "new_algo.py"

        bad_py = edit_file(
            "new_algo.py",
            "",
            "def f(:\n    return 1\n",
            workspace_dir=self.workspace,
        )
        self.assertFalse(bad_py["success"])
        self.assertIn("SyntaxError", bad_py["error"])
        self.assertFalse(new_py.exists())

        with patch("tools.CodeTools.write_file", wraps=write_file) as spy_write:
            created = edit_file(
                "new_algo.py",
                "",
                "def f():\n    return 1\n",
                workspace_dir=self.workspace,
            )
        self.assertTrue(created["success"])
        self.assertTrue(created["created"])
        spy_write.assert_called_once_with(str(new_py), "def f():\n    return 1\n")
        self.assertEqual(new_py.read_text(encoding="utf-8"), "def f():\n    return 1\n")

        # Bezpośrednie write_file z FileTools również waliduje składnię kodu.
        direct_bad = write_file(str(self.workspace / "bad_direct.py"), "def broken(:\n")
        self.assertFalse(direct_bad["success"])
        self.assertIn("SyntaxError", direct_bad["error"])
        self.assertFalse((self.workspace / "bad_direct.py").exists())


if __name__ == "__main__":
    unittest.main()
