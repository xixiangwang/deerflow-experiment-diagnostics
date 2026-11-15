import copy
import importlib.util
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("langchain_core") is not None
if AVAILABLE:
    from deerflow_case import multistep_tools as tools


@unittest.skipUnless(AVAILABLE, "Requires integration dependencies")
class MultiStepStateTests(unittest.TestCase):
    def setUp(self):
        tools.bind_trial("trial-201")
        tools.STATE["last"] = {"status": "failed", "run_id": "initial", "revision": 0,
                               "available_linear_modules": ["q_proj", "v_proj", "lm_head"]}

    def fake_run(self):
        result = {"status": "failed", "revision": tools.STATE["revision"], "run_id": "second",
                  "available_linear_modules": ["q_proj", "v_proj"]}
        tools.STATE["last"] = copy.deepcopy(result)
        return result

    def test_stale_revision_does_not_run_or_mutate(self):
        with patch.object(tools, "execute_current") as run:
            result = tools.apply_trial_repair.invoke({"expected_revision": 7, "repair": "enable_grad"})
            self.assertEqual(result["status"], "stale_revision")
            self.assertFalse(tools.STATE["patches"])
            run.assert_not_called()

    def test_previous_patches_survive_next_repair(self):
        with patch.object(tools, "execute_current", side_effect=self.fake_run):
            tools.apply_trial_repair.invoke({"expected_revision": 0, "repair": "set_target_modules", "target_modules": ["q_proj"]})
            tools.apply_trial_repair.invoke({"expected_revision": 1, "repair": "enable_grad"})
        self.assertEqual(tools.STATE["patches"], ["set_target_modules", "enable_grad"])
        self.assertEqual(tools.STATE["revision"], 2)

    def test_duplicate_repair_uses_cache(self):
        tools.STATE["patches"] = ["enable_grad"]
        with patch.object(tools, "execute_current") as run:
            result = tools.apply_trial_repair.invoke({"expected_revision": 0, "repair": "enable_grad"})
            self.assertTrue(result["cache_hit"])
            run.assert_not_called()

    def test_passed_trial_cannot_be_mutated(self):
        tools.STATE["last"]["status"] = "passed"
        with patch.object(tools, "execute_current") as run:
            result = tools.apply_trial_repair.invoke({"expected_revision": 0, "repair": "pad_sequences"})
            self.assertEqual(result["status"], "already_verified")
            self.assertFalse(tools.STATE["patches"])
            run.assert_not_called()

    def test_worker_infrastructure_error_rolls_back_state(self):
        before = copy.deepcopy(tools.STATE)
        with patch.object(tools, "execute_current", side_effect=RuntimeError("worker unavailable")):
            with self.assertRaises(RuntimeError):
                tools.apply_trial_repair.invoke({"expected_revision": 0, "repair": "enable_grad"})
        self.assertEqual(tools.STATE, before)

    def test_changed_error_is_not_a_successful_trial(self):
        from run_multistep import grade_trial
        trace = [{"result": {"execution_started": True, "status": "failed", "stage": "first", "error": "a"}},
                 {"result": {"execution_started": True, "status": "failed", "stage": "second", "error": "b"}}]
        result = grade_trial({"status": "verified_repair"}, trace)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["multiple_failures_observed"])

    def test_model_must_reference_the_actual_final_run(self):
        from run_multistep import grade_trial
        trace = [{"result": {"execution_started": True, "status": "failed", "stage": "first", "error": "a"}},
                 {"result": {"execution_started": True, "status": "failed", "stage": "second", "error": "b"}},
                 {"result": {"execution_started": True, "status": "passed", "exit_code": 0,
                             "finite_gradients": True, "parameters_changed": True, "run_id": "actual", "revision": 2}}]
        result = grade_trial({"status": "verified_repair", "final_run_id": "invented", "final_revision": 2}, trace)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["structured_final_agrees"])

    def test_parallel_old_revision_can_only_commit_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with patch.object(tools, "execute_current", side_effect=self.fake_run) as run:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(tools.apply_trial_repair.invoke,
                    {"expected_revision": 0, "repair": repair}) for repair in ("enable_grad", "pad_sequences")]
                results = [future.result() for future in futures]
        self.assertEqual({result["status"] for result in results}, {"failed", "stale_revision"})
        self.assertEqual(tools.STATE["revision"], 1)
        self.assertEqual(len(tools.STATE["patches"]), 1)
        self.assertEqual(run.call_count, 1)
