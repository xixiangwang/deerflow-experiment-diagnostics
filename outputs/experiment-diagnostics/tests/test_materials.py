import json
from pathlib import Path
import tempfile
import unittest
from experiment_diagnostics.materials import load_materials, normalize, inspect_materials


class MaterialTests(unittest.TestCase):
    def test_nested_credentials_redacted_but_token_budget_kept(self):
        result = normalize({"log": "Authorization: Bearer abcdef1234", "experiment_config": {
            "api_key": "private-value", "nested": {"access_token": "secret-value"}, "max_tokens": 400}})
        self.assertNotIn("private-value", str(result))
        self.assertNotIn("abcdef1234", str(result))
        self.assertEqual(result["experiment_config"]["max_tokens"], 400)

    def test_submitted_python_is_not_executed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sentinel = root / "executed.txt"
            (root / "code.py").write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')", encoding="utf-8")
            (root / "log.txt").write_text("header\nFileNotFoundError: data.json", encoding="utf-8")
            result = inspect_materials(load_materials(root))
            self.assertFalse(sentinel.exists())
            self.assertFalse(result["validation_executed"])
            self.assertEqual(result["findings"][0]["evidence"][0]["line"], 2)

    def test_only_supported_folder_files_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "log.txt").write_text("hello", encoding="utf-8")
            (root / "apikey.txt").write_text("DO_NOT_READ", encoding="utf-8")
            self.assertNotIn("DO_NOT_READ", str(load_materials(root)))

    def test_sparse_and_malformed_inputs(self):
        self.assertTrue(inspect_materials({"materials": normalize({}), "sources": []})["missing"])
        for value in ({"unexpected": 1}, {"code": 5}, {"environment": {"versions": {"torch": 2}}}):
            with self.assertRaises(ValueError):
                normalize(value)

    def test_file_size_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text("a" * 300001, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_materials(path)

    def test_redaction_preserves_log_lines_and_prefixed_keys(self):
        result = normalize({"log": "Bearer\nheader\nFileNotFoundError", "experiment_config": {
            "AWS_SECRET_ACCESS_KEY": "must-not-leak", "input_tokens": 120}})
        self.assertEqual(len(result["log"].splitlines()), 3)
        self.assertNotIn("must-not-leak", str(result))
        self.assertEqual(result["experiment_config"]["input_tokens"], 120)

    def test_nonfinite_json_config_rejected(self):
        with self.assertRaises(ValueError):
            normalize({"experiment_config": {"learning_rate": float("nan")}})
