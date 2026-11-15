import unittest
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.model_output import material_contract_errors, normalize_material_answer


class ModelOutputTests(unittest.TestCase):
    def test_strict_object_remains_strict(self):
        result = parse_model_object('{"status":"not_executed"}')
        self.assertTrue(result["strict_json"])
        self.assertFalse(result["recovered"])

    def test_markdown_recovery_is_explicit(self):
        result = parse_model_object('说明\n```json\n{"nested":{"number":2}}\n```')
        self.assertEqual(result["value"], {"nested": {"number": 2}})
        self.assertFalse(result["strict_json"])
        self.assertTrue(result["recovered"])

    def test_two_objects_are_not_silently_selected(self):
        result = parse_model_object('{"status":"failed"}\n{"status":"passed"}')
        self.assertIsNone(result["value"])
        self.assertEqual(result["error"], "ambiguous_objects")

    def test_invalid_output_and_nonobject(self):
        for text in ('{"incomplete":', '[1,2]', 'no JSON'):
            self.assertIsNone(parse_model_object(text)["value"])

    def test_braces_inside_json_string_do_not_create_an_extra_object(self):
        result = parse_model_object('text\n{"log":"braces { are data }"}\nend')
        self.assertEqual(result["value"]["log"], 'braces { are data }')

    def test_nested_success_cannot_escape_a_broken_outer_envelope(self):
        result = parse_model_object('{"incomplete": {"status":"verified_repair"}')
        self.assertIsNone(result["value"])
        self.assertEqual(result["error"], "broken_json_envelope")

    def test_material_report_requires_all_typed_sections(self):
        errors = material_contract_errors({"status": "insufficient_evidence", "validation_status": "not_executed"})
        self.assertIn("observations must be an array", errors)
        self.assertIn("limitations must be text or an array", errors)

    def test_complete_readonly_material_report_is_accepted(self):
        answer = {field: [] for field in ("observations", "hypotheses", "recommendations", "verification_plan", "missing_information")}
        answer.update(status="diagnostic_hypotheses", validation_status="not_executed", limitations="未验证")
        self.assertEqual(material_contract_errors(answer), [])
        answer["validation_status"] = "verified"
        self.assertTrue(material_contract_errors(answer))

    def test_only_nonempty_plan_string_is_wrapped_and_execution_claim_is_not_corrected(self):
        raw = {"verification_plan": "核对首批输入。", "validation_status": "verified"}
        answer, fields = normalize_material_answer(raw)
        self.assertEqual(fields, ["verification_plan"])
        self.assertEqual(answer["verification_plan"], [raw["verification_plan"]])
        self.assertEqual(raw["verification_plan"], "核对首批输入。")
        self.assertIn("read-only analysis cannot claim execution", material_contract_errors(answer))
        self.assertEqual(normalize_material_answer({"verification_plan": " "})[1], [])
