import copy
import importlib.util
import unittest
from unittest.mock import patch
from deerflow_case.recipe_spec import normalize_recipe, recipe_hash, apply_change
AVAILABLE = importlib.util.find_spec("langchain_core") is not None
if AVAILABLE:
    from deerflow_case import recipe_tools as tools
    from deerflow_case.recipe_runtime import RecipePolicy, is_worker_report
    from run_recipe import grade_recipe, verified_worker_report


def sample():
    return {"schema_version": 1, "profile": "tiny-qwen2-cpu", "data": {"input_ids": [[2, 3, 4], [4, 5, 6, 7]]}}


class RecipeContractTests(unittest.TestCase):
    def test_no_source_commands_paths_or_extra_fields(self):
        for field in ("code", "command", "model_path", "python", "api_key"):
            with self.assertRaises(ValueError):
                normalize_recipe({**sample(), field: "unexpected"})

    def test_resource_limits_and_numeric_types_are_enforced(self):
        for training in ({"steps": 500}, {"steps": True}, {"learning_rate": float("nan")}, {"gradients_enabled": 1}):
            with self.assertRaises(ValueError):
                normalize_recipe({**sample(), "training": training})
        with self.assertRaises(ValueError):
            normalize_recipe({**sample(), "model": {"vocab_size": 500000}})
        with self.assertRaises(ValueError):
            normalize_recipe({**sample(), "data": {"input_ids": [[1] * 65]}})

    def test_nonexistent_identifier_is_valid_data_not_a_prejudged_error(self):
        recipe = normalize_recipe({**sample(), "lora": {"target_modules": ["not_existing"]}})
        self.assertEqual(recipe["lora"]["target_modules"], ["not_existing"])
        with self.assertRaises(ValueError):
            normalize_recipe({**sample(), "lora": {"target_modules": ["foo;python"]}})

    def test_causal_label_replacement_is_explicit_and_preserves_original(self):
        initial = normalize_recipe({**sample(), "data": {"input_ids": [[2, 3, 4]], "labels": [[2, 3, 75]]}})
        updated = apply_change(initial, "rebuild_causal_labels")
        self.assertIsNone(updated["data"]["labels"])
        self.assertEqual(initial["data"]["labels"], [[2, 3, 75]])
        self.assertNotEqual(recipe_hash(updated), recipe_hash(initial))

    def test_only_observed_internal_target_names_can_be_applied(self):
        initial = normalize_recipe(sample())
        for names in (["lm_head"], ["missing"]):
            with self.assertRaises(ValueError):
                apply_change(initial, "set_lora_targets", names, ["q_proj", "lm_head"])


@unittest.skipUnless(AVAILABLE, "Requires integration dependencies")
class RecipeStateTests(unittest.TestCase):
    def setUp(self):
        tools.bind_recipe(normalize_recipe({**sample(), "training": {"collate": "stack", "gradients_enabled": False}}))
        tools.STATE["last"] = {"status": "failed", "available_linear_modules": ["q_proj", "v_proj"]}

    def observe(self):
        result = {"status": "failed", "available_linear_modules": ["q_proj", "v_proj"], "execution_started": True,
                  "revision": tools.STATE["revision"], "recipe_sha256": recipe_hash(tools.STATE["current"])}
        tools.STATE["last"] = result
        tools.STATE["executions"] += 1
        return result

    def test_cumulative_changes_do_not_reset_previous_repairs(self):
        with patch.object(tools, "execute_current", side_effect=self.observe):
            tools.change_recipe.invoke({"expected_revision": 0, "change": "pad_batch"})
            tools.change_recipe.invoke({"expected_revision": 1, "change": "enable_gradients"})
        self.assertEqual(tools.STATE["current"]["training"]["collate"], "pad")
        self.assertTrue(tools.STATE["current"]["training"]["gradients_enabled"])
        self.assertEqual(tools.STATE["revision"], 2)

    def test_stale_and_rejected_changes_do_not_execute(self):
        with patch.object(tools, "execute_current") as execute:
            stale = tools.change_recipe.invoke({"expected_revision": 1, "change": "pad_batch"})
            invalid = tools.change_recipe.invoke({"expected_revision": 0, "change": "set_lora_targets", "target_modules": ["lm_head"]})
            self.assertEqual(stale["status"], "stale_revision")
            self.assertEqual(invalid["status"], "rejected")
            execute.assert_not_called()

    def test_infrastructure_failure_rolls_back_configuration(self):
        before = copy.deepcopy(tools.STATE)
        with patch.object(tools, "execute_current", side_effect=RuntimeError("infrastructure")):
            with self.assertRaises(RuntimeError):
                tools.change_recipe.invoke({"expected_revision": 0, "change": "pad_batch"})
        self.assertEqual(tools.STATE, before)

    def test_passing_recipe_is_immutable(self):
        tools.STATE["last"].update(status="passed", run_id="actual")
        with patch.object(tools, "execute_current") as execute:
            result = tools.change_recipe.invoke({"expected_revision": 0, "change": "pad_batch"})
            self.assertEqual(result["status"], "already_verified")
            execute.assert_not_called()

    def test_operator_can_forbid_label_replacement_without_mutation_or_execution(self):
        recipe = normalize_recipe({**sample(), "data": {"input_ids": [[2, 3, 4]], "labels": [[2, 3, 75]]}})
        tools.bind_recipe(recipe, allow_label_rebuild=False)
        tools.STATE["last"] = {"status": "failed", "stage": "forward"}
        with patch.object(tools, "execute_current") as execute:
            result = tools.change_recipe.invoke({"expected_revision": 0, "change": "rebuild_causal_labels"})
            execute.assert_not_called()
        self.assertEqual(result["status"], "label_policy_denied")
        self.assertEqual(tools.STATE["current"], recipe)
        self.assertEqual(tools.STATE["revision"], 0)

    def test_duplicate_searches_consume_budget_without_retrieval_and_new_revision_can_search(self):
        with patch.object(tools, "retrieve", return_value={"status": "retrieved", "hits": []}) as retrieve:
            search = tools.search_recipe_reference.invoke
            search({"query": "LoRA modules", "package": "peft"})
            duplicate = search({"query": "  lora  MODULES  ", "package": "peft"})
            self.assertEqual(duplicate["status"], "search_already_attempted")
            self.assertEqual(search({"query": "Another query", "package": "peft"})["status"], "search_budget_exhausted")
            retrieve.assert_called_once()
            with patch.object(tools, "execute_current", side_effect=self.observe):
                tools.change_recipe.invoke({"expected_revision": 0, "change": "pad_batch"})
            self.assertEqual(search({"query": "LoRA modules", "package": "peft"})["status"], "retrieved")
            self.assertEqual(retrieve.call_count, 2)

    def test_worker_label_failure_terminates_with_current_execution_identity(self):
        recipe = normalize_recipe({**sample(), "data": {"input_ids": [[2, 3, 4]], "labels": [[-100, -100, -100]]}})
        tools.bind_recipe(recipe, allow_label_rebuild=False)
        run = {"status": "failed", "stage": "forward", "device": "cpu", "exit_code": 1,
               "label_problem": {"code": "no_supervised_targets"}, "allow_label_rebuild": False,
               "submitted_code_executed": False, "execution_started": True, "changes": [], "recipe": recipe,
               "revision": 0, "run_id": "failed-run", "recipe_sha256": recipe_hash(recipe)}
        tools.STATE["last"] = run
        with patch.object(tools, "execute_current") as execute:
            result = tools.change_recipe.invoke({"expected_revision": 0, "change": "enable_gradients"})
            execute.assert_not_called()
        self.assertEqual(result["status"], "blocked_by_label_policy")
        update = RecipePolicy().before_model({}, None)
        self.assertEqual(update["jump_to"], "end")
        self.assertTrue(is_worker_report(update["messages"][0].model_dump()))
        import json
        answer = json.loads(update["messages"][0].content)
        self.assertTrue(grade_recipe(answer, [{"result": run}])["passed"])
        answer["final_run_id"] = "another-run"
        self.assertFalse(grade_recipe(answer, [{"result": run}])["passed"])
        tools.STATE["allow_label_rebuild"] = True
        self.assertIsNone(RecipePolicy().before_model({}, None))

    def test_report_must_agree_with_real_hash_revision_and_execution(self):
        run = {"status": "passed", "exit_code": 0, "execution_started": True, "revision": 0, "run_id": "actual",
               "profile": "tiny-qwen2-cpu", "device": "cpu", "pretrained_weights_used": False,
               "submitted_code_executed": False, "validation_steps": 2,
               "recipe_sha256": "actual-hash", "finite_gradients": True, "finite_parameters": True,
               "parameters_changed": True, "base_parameters_frozen": True}
        answer = {"status": "verified_baseline", "final_run_id": "actual", "final_revision": 0,
                  "final_recipe_sha256": "actual-hash", "observations": [], "changes": [], "limitations": [], "label_policy_changed": False}
        trace = [{"result": run}]
        self.assertTrue(grade_recipe(answer, trace)["passed"])
        answer["final_recipe_sha256"] = "invented"
        self.assertFalse(grade_recipe(answer, trace)["passed"])
        run["changes"] = []
        self.assertEqual(verified_worker_report(trace)["final_recipe_sha256"], "actual-hash")
        run["finite_gradients"] = False
        self.assertIsNone(verified_worker_report(trace))
        run["finite_gradients"] = True
        run["status"] = "failed"
        self.assertFalse(grade_recipe(answer, trace)["passed"])
        self.assertIsNone(verified_worker_report(trace))
