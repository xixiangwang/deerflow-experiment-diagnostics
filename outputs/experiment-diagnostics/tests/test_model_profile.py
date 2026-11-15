import json
from pathlib import Path
import tempfile
import unittest
from deerflow_case.model_profile import EXPECTED, inspect_local_profile


class ModelProfileTests(unittest.TestCase):
    def profile(self, directory, **changes):
        config = {**EXPECTED, "architectures": ["Qwen2ForCausalLM"], **changes}
        (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (Path(directory) / "model.safetensors").write_bytes(b"metadata-test-only" * 100)

    def test_profile_check_does_not_claim_to_load_or_verify_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            self.profile(directory)
            report = inspect_local_profile(directory)
            self.assertFalse(report["weight_contents_verified"])
            self.assertFalse(report["pretrained_weights_loaded_by_preflight"])

    def test_larger_model_dimensions_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            self.profile(directory, hidden_size=3072, num_hidden_layers=32)
            with self.assertRaisesRegex(ValueError, "Only the Qwen2.5-0.5B"):
                inspect_local_profile(directory)

    def test_custom_code_and_missing_safetensors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.profile(directory, auto_map={"AutoModel": "custom.Model"})
            with self.assertRaisesRegex(ValueError, "Custom model"):
                inspect_local_profile(directory)
            self.profile(directory)
            (Path(directory) / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "safetensors"):
                inspect_local_profile(directory)
