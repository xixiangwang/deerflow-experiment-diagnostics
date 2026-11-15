import ast
import importlib.util
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "deploy/whu/install_bundle.py"
spec = importlib.util.spec_from_file_location("whu_install_contract", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WhuBundleTests(unittest.TestCase):
    def test_installer_is_parseable_by_system_python_39(self):
        ast.parse(PATH.read_text(encoding="utf-8"), feature_version=(3, 9))

    def test_archive_paths_and_link_parents_cannot_escape(self):
        for value in ("../escape", "/absolute", "C:/foreign"):
            with self.assertRaises(ValueError):
                module.validate_members([tarfile.TarInfo(value)])
        link = tarfile.TarInfo("python/lib")
        link.type, link.linkname = tarfile.SYMTYPE, "../outside"
        with self.assertRaises(ValueError):
            module.validate_members([link, tarfile.TarInfo("python/lib/entry")])

    def test_relative_python_executable_link_is_accepted(self):
        link = tarfile.TarInfo("python/bin/python3")
        link.type, link.linkname = tarfile.SYMTYPE, "python3.12"
        module.validate_members([link, tarfile.TarInfo("python/bin/python3.12")])

    def test_slurm_templates_use_lf_no_bom_and_one_gpu_only(self):
        for path in (ROOT / "deploy/whu").glob("*.sbatch"):
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"#!/bin/bash\n"))
            self.assertNotIn(b"\r", raw)
            self.assertNotIn(b"--exclusive", raw)
        setup = (ROOT / "deploy/whu/whu-setup.sbatch").read_text(encoding="utf-8")
        gpu = (ROOT / "deploy/whu/whu-gpu-probe.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("--gres", setup)
        self.assertIn("--gres=gpu:1", gpu)
        self.assertIn("--partition=a100x4", gpu)
