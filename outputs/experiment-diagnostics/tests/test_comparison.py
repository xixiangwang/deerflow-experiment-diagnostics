import importlib.util
import unittest

AVAILABLE = importlib.util.find_spec("langchain_openai") is not None
if AVAILABLE:
    from compare_materials import score


@unittest.skipUnless(AVAILABLE, "Requires API integration dependencies")
class ComparisonScoreTests(unittest.TestCase):
    def test_verified_claim_is_not_accepted_in_readonly_route(self):
        case = {"expected_status": "diagnostic_hypotheses", "coverage_terms": ["梯度"]}
        answer = {"status": "diagnostic_hypotheses", "observations": ["梯度缺失"], "validation_status": "verified"}
        self.assertFalse(score(answer, case)["passed"])

    def test_generic_verification_checklist_is_not_diagnostic_coverage(self):
        case = {"expected_status": "diagnostic_hypotheses", "coverage_terms": ["梯度"]}
        answer = {"status": "diagnostic_hypotheses", "observations": [], "hypotheses": [],
                  "verification_plan": ["检查梯度"], "validation_status": "not_executed"}
        self.assertFalse(score(answer, case)["passed"])

    def test_missing_evidence_requires_request_for_information(self):
        case = {"expected_status": "insufficient_evidence", "coverage_terms": []}
        answer = {"status": "insufficient_evidence", "validation_status": "not_executed", "missing_information": []}
        self.assertFalse(score(answer, case)["passed"])
