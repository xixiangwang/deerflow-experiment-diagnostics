import unittest
from experiment_diagnostics.citations import audit_material_citations


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.hit = {"citation": "https://github.com/official/repo/blob/v1/doc.md#L4-L7", "id": "doc-4",
            "source_sha256": "digest", "source_line_start": 4, "source_line_end": 7,
            "package": "peft", "version": "1.0", "applicability": "version_not_verified"}
        self.trace = [{"tool": "search_material_references", "result": {"hits": [self.hit]}}]

    def test_matching_source_preserves_unverified_version_and_semantic_limits(self):
        result = audit_material_citations({"references": [{"citation": self.hit["citation"], "used_for": "配置说明"}]}, self.trace)
        self.assertTrue(result["passed"])
        self.assertEqual(result["unverified_version_citations"], [self.hit["citation"]])
        self.assertFalse(result["claim_entailment_checked"])
        self.assertEqual(result["matched_references"][0]["source_sha256"], "digest")

    def test_same_document_with_fabricated_lines_is_rejected(self):
        reference = {"citation": self.hit["citation"].replace("L4-L7", "L1-L99"), "used_for": "配置说明"}
        result = audit_material_citations({"references": [reference]}, self.trace)
        self.assertFalse(result["passed"])
        self.assertFalse(result["matched_references"])

    def test_a_previous_or_different_sessions_citation_is_rejected(self):
        result = audit_material_citations({"references": [{"citation": self.hit["citation"], "used_for": "说明"}]}, [])
        self.assertFalse(result["passed"])

    def test_missing_and_duplicate_references_do_not_pass(self):
        self.assertFalse(audit_material_citations({}, self.trace)["passed"])
        reference = {"citation": self.hit["citation"], "used_for": "说明"}
        self.assertFalse(audit_material_citations({"references": [reference, reference]}, self.trace)["passed"])
        self.assertTrue(audit_material_citations({"references": []}, self.trace)["passed"])


if __name__ == "__main__":
    unittest.main()
