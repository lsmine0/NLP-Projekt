import os
from datetime import datetime, timezone
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


from tools.FileTools import (
    create_file,
    read_file,
    write_file,
    move_file,
    delete_file,
    list_directory,
    search_in_files,
    get_file_info,
    find_files 
)

class FileToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.directory = Path(self.temp_directory.name)
        self.file_path = self.directory / "example.txt"

    def test_create_and_read_file(self):
        content = "Przykładowa treść: ąęłńóśźż."

        result = create_file(str(self.file_path), content)

        self.assertTrue(result["success"])
        self.assertEqual(
            self.file_path.read_text(encoding="utf-8"),
            content,
        )

        result = read_file(str(self.file_path))

        self.assertTrue(result["success"])
        self.assertEqual(result["content"], content)

    def test_create_empty_file(self):
        result = create_file(str(self.file_path))

        self.assertTrue(result["success"])
        self.assertEqual(
            self.file_path.read_text(encoding="utf-8"),
            "",
        )

    def test_create_does_not_overwrite(self):
        self.file_path.write_text("Oryginalna treść", encoding="utf-8")

        result = create_file(str(self.file_path), "Nowa treść")

        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(
            self.file_path.read_text(encoding="utf-8"),
            "Oryginalna treść",
        )

    def test_write_file(self):
        self.file_path.write_text("Stara treść", encoding="utf-8")

        result = write_file(str(self.file_path), "Nowa treść")

        self.assertTrue(result["success"])
        self.assertEqual(
            self.file_path.read_text(encoding="utf-8"),
            "Nowa treść",
        )

    def test_move_file(self):
        self.file_path.write_text("Treść pliku", encoding="utf-8")
        target_directory = self.directory / "destination"
        target_directory.mkdir()
        destination = target_directory / "moved.txt"

        result = move_file(str(self.file_path), str(destination))

        self.assertTrue(result["success"])
        self.assertFalse(self.file_path.exists())
        self.assertEqual(
            destination.read_text(encoding="utf-8"),
            "Treść pliku",
        )

    def test_move_does_not_overwrite(self):
        self.file_path.write_text("Źródło", encoding="utf-8")
        destination = self.directory / "existing.txt"
        destination.write_text("Cel", encoding="utf-8")

        result = move_file(str(self.file_path), str(destination))

        self.assertFalse(result["success"])
        self.assertTrue(self.file_path.exists())
        self.assertEqual(
            destination.read_text(encoding="utf-8"),
            "Cel",
        )

    def test_delete_file(self):
        self.file_path.write_text("Do usunięcia", encoding="utf-8")

        result = delete_file(str(self.file_path))

        self.assertTrue(result["success"])
        self.assertFalse(self.file_path.exists())

    def test_list_directory(self):
        self.file_path.touch()
        (self.directory / "subdirectory").mkdir()

        result = list_directory(str(self.directory))

        self.assertTrue(result["success"])

        entries = {
            entry["name"]: entry["type"]
            for entry in result["entries"]
        }

        self.assertEqual(
            entries,
            {
                "example.txt": "file",
                "subdirectory": "directory",
            },
        )

    def test_read_missing_file(self):
        result = read_file(str(self.file_path))

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_list_file_instead_of_directory(self):
        self.file_path.touch()

        result = list_directory(str(self.file_path))

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_delete_does_not_remove_directory(self):
        result = delete_file(str(self.directory))

        self.assertFalse(result["success"])
        self.assertTrue(self.directory.is_dir())

    def test_search_in_files(self):
        self.file_path.write_text(
            "Pierwsza linia\nSzukana TREŚĆ\nOstatnia linia",
            encoding="utf-8",
        )

        subdirectory = self.directory / "nested"
        subdirectory.mkdir()
        nested_file = subdirectory / "another.txt"
        nested_file.write_text(
            "Tutaj również jest treść.",
            encoding="utf-8",
        )

        result = search_in_files(str(self.directory), "treść")

        self.assertTrue(result["success"])
        self.assertEqual(result["matches_count"], 2)
        self.assertEqual(
            {
                (match["path"], match["line_number"], match["line"])
                for match in result["matches"]
            },
            {
                (str(self.file_path), 2, "Szukana TREŚĆ"),
                (str(nested_file), 1, "Tutaj również jest treść."),
            },
        )

    def test_search_case_sensitive(self):
        self.file_path.write_text("Python\npython", encoding="utf-8")

        result = search_in_files(
            str(self.file_path),
            "Python",
            case_sensitive=True,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["matches_count"], 1)
        self.assertEqual(result["matches"][0]["line_number"], 1)
        self.assertEqual(result["matches"][0]["line"], "Python")

    def test_search_without_matches(self):
        self.file_path.write_text(
            "Przykładowa treść",
            encoding="utf-8",
        )

        result = search_in_files(
            str(self.file_path),
            "nieobecny fragment",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["matches_count"], 0)
        self.assertEqual(result["matches"], [])

    def test_search_empty_text(self):
        result = search_in_files(str(self.directory), "")

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_search_skips_non_utf8_file(self):
        binary_file = self.directory / "binary.bin"
        binary_file.write_bytes(b"\xff\xfe\x00")
        self.file_path.write_text(
            "Szukany tekst",
            encoding="utf-8",
        )

        result = search_in_files(str(self.directory), "Szukany")

        self.assertTrue(result["success"])
        self.assertEqual(result["matches_count"], 1)
        self.assertEqual(len(result["skipped_files"]), 1)
        self.assertEqual(
            result["skipped_files"][0]["path"],
            str(binary_file),
        )
    def test_get_file_info(self):
        content = "Przykładowa treść"
        self.file_path.write_bytes(content.encode("utf-8"))
    
        result = get_file_info(str(self.file_path))
    
        self.assertTrue(result["success"])
        self.assertEqual(result["name"], "example.txt")
        self.assertEqual(result["extension"], ".txt")
        self.assertEqual(result["size_bytes"], len(content.encode("utf-8")))
        self.assertTrue(result["is_file"])
        self.assertFalse(result["is_directory"])
        self.assertTrue(Path(result["path"]).is_absolute())
        self.assertEqual(
            datetime.fromisoformat(result["modified_at"]).tzinfo,
            timezone.utc,
        )
    
    
    def test_get_directory_info(self):
        result = get_file_info(str(self.directory))
    
        self.assertTrue(result["success"])
        self.assertFalse(result["is_file"])
        self.assertTrue(result["is_directory"])
        self.assertIsNone(result["size_bytes"])
    
    
    def test_get_info_missing_file(self):
        result = get_file_info(str(self.file_path))
    
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    
    def test_find_files_filters(self):
        nested = self.directory / "nested"
        nested.mkdir()
    
        old_pdf = nested / "REPORT_old.PDF"
        boundary_pdf = self.directory / "report_boundary.pdf"
        old_text = self.directory / "report_old.txt"
        other_pdf = self.directory / "invoice.pdf"
    
        old_timestamp = datetime(
            2025, 12, 1, tzinfo=timezone.utc
        ).timestamp()
    
        boundary_timestamp = datetime(
            2026, 1, 1, tzinfo=timezone.utc
        ).timestamp()
    
        for file_path in (old_pdf, old_text, other_pdf):
            file_path.touch()
            os.utime(file_path, (old_timestamp, old_timestamp))
    
        boundary_pdf.touch()
        os.utime(
            boundary_pdf,
            (boundary_timestamp, boundary_timestamp),
        )
    
        result = find_files(
            str(self.directory),
            name_pattern="report*",
            extensions=["pdf"],
            modified_before="2026-01-01",
        )
    
        self.assertTrue(result["success"])
        self.assertEqual(result["files_count"], 1)
        self.assertEqual(
            Path(result["files"][0]["path"]),
            old_pdf.absolute(),
        )
    
    
    def test_find_files_without_recursion(self):
        self.file_path.touch()
        nested = self.directory / "nested"
        nested.mkdir()
        (nested / "hidden.txt").touch()
    
        result = find_files(str(self.directory), recursive=False)
    
        self.assertTrue(result["success"])
        self.assertEqual(
            [file["name"] for file in result["files"]],
            ["example.txt"],
        )
    
    
    def test_find_files_without_matches(self):
        self.file_path.touch()
    
        result = find_files(str(self.directory), extensions=[".pdf"])
    
        self.assertTrue(result["success"])
        self.assertEqual(result["files_count"], 0)
        self.assertEqual(result["files"], [])
    
    
    def test_find_files_invalid_date(self):
        result = find_files(
            str(self.directory),
            modified_before="2026-99-99",
        )
    
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    
    def test_find_files_requires_directory(self):
        self.file_path.touch()
    
        result = find_files(str(self.file_path))
    
        self.assertFalse(result["success"])
        self.assertIn("error", result)    


if __name__ == "__main__":
    unittest.main(verbosity=2)
