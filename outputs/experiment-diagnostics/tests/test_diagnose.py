import asyncio
import unittest
from experiment_diagnostics import install, inspect_experiment
from experiment_diagnostics.diagnose import diagnose


class DiagnosisTests(unittest.TestCase):
    def test_unknown_is_not_healthy(self):
        self.assertEqual(diagnose({"log": "training stopped"})["status"], "insufficient_evidence")

    def test_nan_does_not_assert_learning_rate(self):
        f = diagnose({"log": "epoch 2 loss=nan"})["findings"][0]
        self.assertEqual(f["certainty"], "hypothesis")
        self.assertEqual(f["evidence"][0]["line"], 1)

    def test_missing_dependency(self):
        result = diagnose({"log": "header\nModuleNotFoundError: No module named 'torch'"})
        self.assertEqual(result["findings"][0]["evidence"][0]["line"], 2)

    def test_zero_division_needs_config(self):
        self.assertFalse(diagnose({"log": "ZeroDivisionError"})["findings"])

    def test_oom(self):
        self.assertEqual(diagnose({"log": "CUDA out of memory"})["findings"][0]["code"], "cuda_oom")

    def test_bad_inputs(self):
        for payload in ({"log": []}, {"log": "a" * 100001}, {"config": []}):
            with self.assertRaises(ValueError):
                diagnose(payload)

    def test_public_contract_registration(self):
        class Registry:
            def plugin(self, contribution):
                self.contribution = contribution
                return True
        registry = Registry()
        install(registry, {})
        tool = registry.contribution.tools[0]
        result = asyncio.run(tool.handler({"log": "FileNotFoundError: data.csv"}, None))
        self.assertEqual(result["findings"][0]["code"], "missing_file")


if __name__ == "__main__":
    unittest.main()
