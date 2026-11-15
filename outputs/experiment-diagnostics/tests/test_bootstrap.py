import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from bootstrap_runtime import validate_target, ROOT, OWNER, MARKER, clean_environment


class BootstrapSafetyTests(unittest.TestCase):
    def test_existing_unmarked_environment_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "existing-package.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                validate_target(directory)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_source_and_source_ancestor_cannot_be_runtime(self):
        for path in (ROOT, ROOT.parent, ROOT / "runtime"):
            with self.assertRaises(ValueError):
                validate_target(path)

    def test_only_owned_environment_can_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / MARKER
            marker.write_text(json.dumps({"owner": "another-project"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_target(directory)
            marker.write_text(json.dumps({"owner": OWNER}), encoding="utf-8")
            self.assertEqual(validate_target(directory), Path(directory).resolve())

    def test_validation_processes_do_not_receive_model_credentials(self):
        with patch.dict("os.environ", {"DIAG_API_KEY": "fake", "DIAG_ACCESS_TOKEN": "fake", "DEER_FLOW_HOME": "other", "PYTHONPATH": "other"}):
            environment = clean_environment()
            self.assertFalse(any(key.startswith(("DIAG_", "DEER_FLOW_")) for key in environment))
            self.assertNotIn("PYTHONPATH", environment)

    def test_offline_plan_does_not_create_or_install_an_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-runtime"
            wheelhouse = Path(directory) / "wheels"
            wheelhouse.mkdir()
            process = subprocess.run([sys.executable, str(ROOT / "bootstrap_runtime.py"),
                "--venv", str(target), "--wheelhouse", str(wheelhouse)], capture_output=True,
                encoding="utf-8", timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout)
            self.assertIn("--no-index", plan["install_command"])
            self.assertIn("--require-hashes", plan["install_command"])
            self.assertFalse(plan["calls_model_api"])
            self.assertFalse(target.exists())
