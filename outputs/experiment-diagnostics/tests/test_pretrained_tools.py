import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("langchain_core") is not None
if AVAILABLE:
    from deerflow_case import pretrained_tools as tools


@unittest.skipUnless(AVAILABLE, "Requires integration dependencies")
class PretrainedToolTests(unittest.TestCase):
    def setUp(self):
        tools.TRACE.clear()

    def test_no_gpu_execution_before_inspection(self):
        with patch.object(tools, "execute") as execute:
            with self.assertRaises(ValueError):
                tools.verify_pretrained_repair.invoke({"target_modules": ["q_proj"]})
            execute.assert_not_called()

    def test_only_observed_modules_can_be_selected(self):
        tools.TRACE.append({"tool": "inspect_pretrained_trial", "result": {"available_linear_modules": ["q_proj", "v_proj", "lm_head"]}})
        with patch.object(tools, "execute") as execute:
            with self.assertRaises(ValueError):
                tools.verify_pretrained_repair.invoke({"target_modules": ["lm_head"]})
            execute.assert_not_called()

    def test_gpu_retry_budget_is_bounded(self):
        tools.TRACE.extend([{"tool": "inspect_pretrained_trial", "result": {}},
                            {"tool": "verify_pretrained_repair"}, {"tool": "verify_pretrained_repair"}])
        with patch.object(tools, "execute") as execute:
            result = tools.verify_pretrained_repair.invoke({"target_modules": ["q_proj"]})
            self.assertEqual(result["status"], "budget_exhausted")
            execute.assert_not_called()

    def test_repeated_inspection_uses_explicit_cache(self):
        tools.TRACE.append({"tool": "inspect_pretrained_trial", "result": {
            "status": "failed", "available_linear_modules": ["q_proj"]}})
        with patch.object(tools, "execute") as execute:
            result = tools.inspect_pretrained_trial.invoke({})
            self.assertTrue(result["cache_hit"])
            self.assertFalse(result["execution_started"])
            execute.assert_not_called()
