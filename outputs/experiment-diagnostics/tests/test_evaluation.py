"""Ensure model claims cannot replace actual repair evidence in scoring."""
import importlib.util
import unittest

AVAILABLE = importlib.util.find_spec("yaml") is not None
if AVAILABLE:
    from evaluate_deerflow import grade


@unittest.skipUnless(AVAILABLE, "Requires evaluation environment")
class EvaluationEvidenceTests(unittest.TestCase):
    def test_claimed_repair_without_execution_fails(self):
        case = {"id": "exp-103", "expected_status": "verified_repair", "expected_fix": "enable_grad"}
        answer = {"case_id": "exp-103", "status": "verified_repair", "validation": "passed"}
        trace = [{"tool": "inspect_experiment_case", "args": {"case_id": "exp-103"}, "result": {"status": "failed"}}]
        self.assertFalse(grade(case, answer, trace)["passed"])

    def test_healthy_case_with_blocked_repair_attempt_fails(self):
        case = {"id": "exp-105", "expected_status": "no_fault_observed", "expected_fix": None}
        answer = {"case_id": "exp-105", "status": "no_fault_observed"}
        trace = [
            {"tool": "inspect_experiment_case", "args": {"case_id": "exp-105"}, "result": {"status": "passed"}},
            {"tool": "verify_experiment_repair", "args": {"case_id": "exp-105"}, "result": {"status": "not_permitted"}},
        ]
        self.assertFalse(grade(case, answer, trace)["passed"])

    def test_other_case_evidence_is_rejected(self):
        case = {"id": "exp-106", "expected_status": "insufficient_evidence", "expected_fix": None}
        answer = {"case_id": "exp-106", "status": "insufficient_evidence"}
        trace = [{"tool": "inspect_experiment_case", "args": {"case_id": "exp-105"},
                  "result": {"status": "insufficient_evidence"}}]
        self.assertFalse(grade(case, answer, trace)["passed"])

    def test_invalid_model_format_does_not_receive_a_pass(self):
        case = {"id": "exp-106", "expected_status": "insufficient_evidence", "expected_fix": None}
        trace = [{"tool": "inspect_experiment_case", "args": {"case_id": "exp-106"},
                  "result": {"status": "insufficient_evidence"}}]
        self.assertFalse(grade(case, "I fixed it", trace)["passed"])

    def test_repaired_without_retrieval_keeps_protocol_failure(self):
        case = {"id": "exp-103", "expected_status": "verified_repair", "expected_fix": "enable_grad"}
        answer = {"case_id": "exp-103", "status": "verified_repair"}
        trace = [
            {"tool": "inspect_experiment_case", "args": {"case_id": "exp-103"}, "result": {"status": "failed"}},
            {"tool": "verify_experiment_repair", "args": {"case_id": "exp-103", "fix_type": "enable_grad"},
             "result": {"status": "passed", "finite_gradients": True, "parameters_changed": True}},
        ]
        result = grade(case, answer, trace)
        self.assertFalse(result["passed"])
        self.assertFalse(result["protocol_passed"])
        self.assertTrue(result["task_outcome_passed"])
