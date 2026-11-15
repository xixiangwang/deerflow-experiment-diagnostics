"""Rehearse the fixed two-fault interview case on the real CPU worker."""
import json
from pathlib import Path
import time
import uuid

from deerflow_case import recipe_tools as tools
from deerflow_case.recipe_spec import canonical_bytes, load_recipe, recipe_hash
from run_recipe import grade_recipe, verified_worker_report

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "examples/recipes/stack-and-targets.json"


def show(result):
    print(f'配置版本 {result["revision"]} · {result["stage"]} · {result["status"]}', flush=True)
    if result.get("error"):
        print("  " + result["error"], flush=True)


def main():
    initial, _ = load_recipe(CASE)
    tools.bind_recipe(initial, allow_label_rebuild=False)
    print("离线演示使用预设修改步骤，由 CPU worker 实际执行。", flush=True)
    baseline = tools.inspect_recipe.invoke({})
    show(baseline)
    if baseline.get("stage") != "adapter_injection" or baseline.get("status") != "failed":
        raise SystemExit("预期的 LoRA 目标模块故障未复现。")

    targets = [name for name in ("q_proj", "v_proj") if name in baseline["available_linear_modules"]]
    print("根据模型实际模块名，修改 LoRA 目标为 " + ", ".join(targets), flush=True)
    after_targets = tools.change_recipe.invoke({"expected_revision": baseline["revision"],
                                               "change": "set_lora_targets", "target_modules": targets})
    show(after_targets)
    if after_targets.get("stage") != "batch_assembly" or after_targets.get("status") != "failed":
        raise SystemExit("预期的变长批次故障未复现。")

    print("保留 LoRA 修改，将批次组装改为 padding。", flush=True)
    final = tools.change_recipe.invoke({"expected_revision": after_targets["revision"], "change": "pad_batch"})
    show(final)
    answer = verified_worker_report(tools.TRACE)
    grade = grade_recipe(answer, tools.TRACE)
    folder = ROOT / "reports/recipe-demo" / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    report = {"mode": "scripted_cpu_rehearsal", "model_api_called": False, "gpu_used": False,
              "initial_recipe_sha256": recipe_hash(initial), "answer": answer, "grade": grade, "trace": tools.TRACE}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "final-recipe.json").write_bytes(canonical_bytes(tools.STATE["current"]))
    if grade["passed"]:
        print(answer["observations"][0], flush=True)
    print(f"运行回执 {folder / 'report.json'}", flush=True)
    return grade["passed"]


if __name__ == "__main__":
    raise SystemExit(not main())
