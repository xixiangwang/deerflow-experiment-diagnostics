import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources, digest


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("DIAG_KNOWLEDGE_INDEX", None)

    def test_snapshot_survives_change_without_reading_credentials_or_user_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scenario"
            root.mkdir()
            (root / "run.py").write_text("print('old')\n", encoding="utf-8")
            (root / ".env").write_text("PRIVATE_TEST_SENTINEL", encoding="utf-8")
            (root / "submitted.json").write_text('"PRIVATE_TEST_SENTINEL"', encoding="utf-8")
            (root / "knowledge").mkdir()
            (root / "knowledge/index.sqlite3").write_bytes(b"known index bytes")
            folder = Path(directory) / "run"
            manifest = capture_run_sources(folder, "run.py", root=root)
            self.assertTrue(check_run_sources(manifest, root=root)["unchanged"])
            self.assertEqual(manifest["scenario_files"], [{"path": "run.py", "sha256": digest(root / "run.py")}])
            self.assertNotIn("PRIVATE_TEST_SENTINEL", json.dumps(manifest))
            (root / "run.py").write_text("print('new')\n", encoding="utf-8")
            self.assertEqual(check_run_sources(manifest, root=root)["changed_paths"], ["run.py"])
            self.assertEqual((folder / "provenance/sources/run.py").read_text(encoding="utf-8"), "print('old')\n")

    def test_changed_knowledge_index_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scenario"
            (root / "knowledge").mkdir(parents=True)
            index = root / "knowledge/index.sqlite3"
            index.write_bytes(b"first index")
            manifest = capture_run_sources(Path(directory) / "run", "test", root=root)
            index.write_bytes(b"second index")
            self.assertIn("knowledge/index.sqlite3", check_run_sources(manifest, root=root)["changed_paths"])

    def test_operator_selected_saved_index_is_the_one_that_gets_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scenario"
            root.mkdir()
            index = Path(directory) / "selected.sqlite3"
            index.write_bytes(b"selected previous index")
            with patch.dict("os.environ", {"DIAG_KNOWLEDGE_INDEX": str(index)}):
                manifest = capture_run_sources(Path(directory) / "run", "test", root=root)
                self.assertEqual(manifest["knowledge_index_sha256"], digest(index))
                self.assertTrue(check_run_sources(manifest, root=root)["unchanged"])
                index.write_bytes(b"unexpected replacement")
                self.assertFalse(check_run_sources(manifest, root=root)["unchanged"])


if __name__ == "__main__":
    unittest.main()
