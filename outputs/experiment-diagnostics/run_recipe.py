"""Official DeerFlow entry for an explicitly selected, bounded JSON experiment recipe."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid
from deerflow_case.recipe_spec import load_recipe, canonical_bytes, recipe_hash
from deerflow_case.recipe_runtime import REPORT_SOURCE, create_recipe_client, is_worker_report
from deerflow_case.runtime import final_model_text
from experiment_diagnostics.materials import redact
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources

ROOT = Path(__file__).resolve().parent
FALLBACK_SOURCE = "worker_verified_fallback"


def grade_recipe(answer, trace):
    runs = [call["result"] for call in trace if call["result"].get("execution_started")]
    final = runs[-1] if runs else {}
    baseline = bool(runs) and runs[0].get("status") == "passed"
    expected = "verified_baseline" if baseline else "verified_repair"
    labels_rebuilt = any(change.get("change") == "rebuild_causal_labels" for change in final.get("changes", []))
    if (final.get("status") == "failed" and final.get("stage") == "forward"
            and final.get("allow_label_rebuild") is False and final.get("label_problem")):
        checks = {"actual_worker_label_failure": final.get("exit_code") == 1 and final.get("device") == "cpu"
                    and final.get("submitted_code_executed") is False,
            "labels_preserved": not labels_rebuilt and final["recipe"]["data"]["labels"] == runs[0]["recipe"]["data"]["labels"],
            "report_matches_blocked_run": isinstance(answer, dict) and answer.get("status") == "unresolved"
                and answer.get("blocked_by_label_policy") is True and answer.get("label_policy_changed") is False
                and answer.get("label_problem") == final["label_problem"] and answer.get("final_run_id") == final["run_id"]
                and type(answer.get("final_revision")) is int and answer["final_revision"] == final["revision"]
                and answer.get("final_recipe_sha256") == final["recipe_sha256"]}
        return {"passed": all(checks.values()), "checks": checks}
    checks = {"actual_execution": bool(runs),
        "report_contract": isinstance(answer, dict) and all(isinstance(answer.get(key), list) for key in ("observations", "changes", "limitations")),
        "final_execution_passed": final.get("status") == "passed" and final.get("exit_code") == 0,
        "finite_gradients_and_parameters": final.get("finite_gradients") is True and final.get("finite_parameters") is True,
        "adapter_updated_and_base_frozen": final.get("parameters_changed") is True and final.get("base_parameters_frozen") is True,
        "bounded_cpu_profile": final.get("profile") == "tiny-qwen2-cpu" and final.get("device") == "cpu"
            and final.get("pretrained_weights_used") is False and final.get("submitted_code_executed") is False
            and 2 <= final.get("validation_steps", 0) <= 3,
        "label_policy_change_reported": isinstance(answer, dict) and type(answer.get("label_policy_changed")) is bool
            and answer["label_policy_changed"] == labels_rebuilt,
        "report_matches_run": isinstance(answer, dict) and answer.get("status") == expected and
            answer.get("final_run_id") == final.get("run_id") and type(answer.get("final_revision")) is int and answer.get("final_revision") == final.get("revision") and
            answer.get("final_recipe_sha256") == final.get("recipe_sha256"),
        "no_mutation_of_passing_baseline": not baseline or all(run.get("revision") == 0 for run in runs)}
    return {"passed": all(checks.values()), "checks": checks}


def verified_worker_report(trace):
    runs = [call["result"] for call in trace if call["result"].get("execution_started")]
    if not runs or runs[-1].get("status") != "passed":
        return None
    final = runs[-1]
    answer = {"status": "verified_baseline" if final["revision"] == 0 else "verified_repair",
              "observations": [f'CPU worker 完成 {final["validation_steps"]} 步验证。梯度和参数均有限，adapter 已更新，基础参数保持冻结。'],
              "changes": final["changes"], "final_run_id": final["run_id"],
              "final_revision": final["revision"], "final_recipe_sha256": final["recipe_sha256"],
              "label_policy_changed": any(change["change"] == "rebuild_causal_labels" for change in final["changes"]),
              "blocked_by_label_policy": False, "limitations": ["结论依据固定随机微型 Qwen2 的 CPU 执行回执。"]}
    return answer if grade_recipe(answer, trace)["passed"] else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--preserve-labels", action="store_true")
    args = parser.parse_args()
    recipe, input_sha256 = load_recipe(args.input)
    if not os.environ.get("DIAG_TORCH_PYTHON"):
        parser.error("Select an existing torch/peft/transformers interpreter")
    run_id = "recipe-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/recipes" / run_id
    folder.mkdir(parents=True)
    provenance = capture_run_sources(folder, "run_recipe.py")
    (folder / "input-recipe.json").write_bytes(canonical_bytes(recipe))
    from deerflow_case.recipe_tools import bind_recipe, STATE, TRACE
    bind_recipe(recipe, allow_label_rebuild=not args.preserve_labels)
    client = create_recipe_client(run_id)
    events, error = [], None
    try:
        for event in client.stream("/declarative-experiment 诊断当前绑定的配置与小批数据，并根据真实 CPU 结果验证修改。",
            thread_id=run_id, recursion_limit=130):
            events.append({"type": event.type, "data": event.data})
    except Exception as exc:
        error = redact(type(exc).__name__ + ": " + str(exc))
    raw = final_model_text(events)
    parsed = parse_model_object(raw)
    assessment = grade_recipe(parsed["value"], TRACE)
    integrity = check_run_sources(provenance)
    local_report = any(is_worker_report(event["data"]) for event in events if event["type"] == "messages-tuple")
    source = REPORT_SOURCE if local_report else "model"
    raw_format = {k: v for k, v in parsed.items() if k != "value"}
    delivered = raw
    if not assessment["passed"] and error is None and integrity["unchanged"]:
        fallback = verified_worker_report(TRACE)
        if fallback is not None:
            source = FALLBACK_SOURCE
            delivered = json.dumps(fallback, ensure_ascii=False)
            parsed = parse_model_object(delivered)
            assessment = grade_recipe(fallback, TRACE)
    assessment["passed"] = assessment["passed"] and error is None and integrity["unchanged"]
    report = {"run_id": run_id, "recipe_protocol": 1, "deerflow_version": "2.0.0", "real_llm_used": True,
        "report_source": source, "preserve_labels": args.preserve_labels,
        "model": os.environ.get("DIAG_MODEL"), "initial_recipe_sha256": recipe_hash(recipe),
        "input_transport_sha256": input_sha256,
        "answer": parsed["value"], "raw_answer": raw, "format": raw_format,
        "strict_output_passed": source != FALLBACK_SOURCE and assessment["passed"] and raw_format["strict_json"],
        "delivered_strict_json": parsed["strict_json"],
        "grade": assessment, "error": error, "trace": TRACE, "events": events,
        "source_integrity": integrity, "submitted_code_executed": False, "cpu_recipe_executed": bool(STATE["executions"]),
        "scope": "操作者选定的受限 JSON 配置与小批数据；固定随机微型 CPU 模型重现，不是任意用户代码、预训练模型质量或独立准确率。"}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "answer.md").write_text(delivered, encoding="utf-8")
    (folder / "final-recipe.json").write_bytes(canonical_bytes(STATE["current"]))
    (ROOT / "reports/latest-recipe.json").write_text(json.dumps({"folder": str(folder), "passed": assessment["passed"]}, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), "passed": assessment["passed"], "strict_output_passed": report["strict_output_passed"]}))
    if not assessment["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
