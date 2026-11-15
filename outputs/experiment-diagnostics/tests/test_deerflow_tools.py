"""Optional adapter tests; run in work/check-env alongside the harness test."""
import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("langchain_core") is not None
if AVAILABLE:
    from deerflow_case.tools import inspect_lora_experiment, validate_lora_fix
    from deerflow_case.batch_tools import verify_experiment_repair


@unittest.skipUnless(AVAILABLE, "Requires the DeerFlow integration environment")
class DeerFlowToolBoundaryTests(unittest.TestCase):
    def test_unknown_case_never_starts_process(self):
        with patch("deerflow_case.tools.worker") as worker:
            with self.assertRaises(ValueError):
                inspect_lora_experiment.invoke({"case_id": "user_supplied_script"})
            worker.assert_not_called()

    def test_unobserved_or_code_like_module_rejected(self):
        for targets in (["exec('bad')"], ["lm_head"], [], ["q_proj"] * 9):
            with patch("deerflow_case.tools.worker") as worker:
                with self.assertRaises(ValueError):
                    validate_lora_fix.invoke({"case_id": "tiny_qwen2_lora", "target_modules": targets})
                worker.assert_not_called()

    def test_no_repair_process_for_healthy_or_missing_evidence_case(self):
        for case_id in ("exp-105", "exp-106"):
            with patch("deerflow_case.batch_tools.execute") as execute:
                result = verify_experiment_repair.invoke({"case_id": case_id, "fix_type": "enable_grad"})
                self.assertEqual(result["status"], "not_permitted")
                self.assertFalse(result["execution_started"])
                execute.assert_not_called()

    def test_material_route_blocks_execution_tools(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from deerflow_case.material_policy import ReadOnlyMaterialPolicy
        handler = Mock()
        result = ReadOnlyMaterialPolicy().wrap_tool_call(SimpleNamespace(
            tool_call={"name": "bash", "id": "blocked", "args": {"command": "echo no"}}), handler)
        self.assertIn("not permitted", result.content)
        handler.assert_not_called()
