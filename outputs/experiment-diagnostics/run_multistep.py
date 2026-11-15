"""Actual DeerFlow/DeepSeek multi-fault diagnosis with real cumulative CPU reruns."""
import json
import argparse
import os
from pathlib import Path
import time
import uuid
from experiment_diagnostics.materials import redact
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources
from deerflow_case.runtime import create_api_client, final_model_text

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent / "work"


def grade_trial(answer, trace):
    runs = [entry["result"] for entry in trace if entry["result"].get("execution_started")]
    failed = [run for run in runs if run.get("status") == "failed"]
    successful = [run for run in runs if run.get("status") == "passed"]
    final = runs[-1] if runs else {}
    checks = {
        "real_initial_failure": bool(runs) and runs[0].get("status") == "failed",
        "multiple_failures_observed": len({(run.get("stage"), run.get("error")) for run in failed}) >= 2,
        "final_execution_passed": final.get("status") == "passed" and final.get("exit_code") == 0,
        "gradient_and_update_proof": final.get("finite_gradients") is True and final.get("parameters_changed") is True,
        "structured_final_agrees": isinstance(answer, dict) and answer.get("status") == "verified_repair"
            and answer.get("final_run_id") == final.get("run_id") and answer.get("final_revision") == final.get("revision"),
        "no_failed_run_after_success": not successful or successful[-1] is final,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=["trial-201", "trial-202", "trial-203"])
    args = parser.parse_args()
    missing = [key for key in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL", "DIAG_TORCH_PYTHON") if not os.environ.get(key)]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    run_id = "multistep-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/multistep" / run_id
    folder.mkdir(parents=True)
    provenance = capture_run_sources(folder, "run_multistep.py")
    from deerflow_case.multistep_tools import bind_trial, TRACE
    client = create_api_client(run_id, "deerflow_case.multistep_tools",
        ["inspect_trial", "apply_trial_repair", "lookup_trial_reference"], "multistep-experiment", 10)
    results = []
    trials = [args.trial] if args.trial else ["trial-201", "trial-202", "trial-203"]
    for trial in trials:
        bind_trial(trial)
        events, error = [], None
        start = time.perf_counter()
        print("BEGIN", trial, flush=True)
        try:
            for event in client.stream("/multistep-experiment 排查并验证当前绑定的实验 " + trial +
                    "。根据实际执行反馈决定下一步，不要在错误仅发生变化时宣称成功。",
                    thread_id=run_id + "-" + trial, recursion_limit=130):
                events.append({"type": event.type, "data": event.data})
        except Exception as exc:
            error = redact(type(exc).__name__ + ": " + str(exc))
        text = final_model_text(events)
        parsed_output = parse_model_object(text)
        answer = parsed_output["value"]
        assessment = grade_trial(answer, TRACE)
        source_integrity = check_run_sources(provenance)
        if error or not source_integrity["unchanged"]:
            assessment["passed"] = False
        record = {"trial": trial, "real_llm_used": True, "deerflow_version": "2.0.0",
                  "model": os.environ["DIAG_MODEL"], "answer": answer, "raw_answer": text,
                  "output_format": {key: value for key, value in parsed_output.items() if key != "value"},
                  "strict_output_passed": assessment["passed"] and parsed_output["strict_json"],
                  "trace": list(TRACE), "events": events, "error": error, "grade": assessment,
                  "provenance": {"manifest": "provenance/manifest.json", "source_integrity": source_integrity},
                  "elapsed_seconds": time.perf_counter() - start,
                  "scope": "预设多故障开发实验；每轮一个预定义改变，真实 CPU 重跑；非任意用户实验和泛化准确率。"}
        (folder / (trial + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        results.append({"trial": trial, "grade": assessment, "strict_output_passed": record["strict_output_passed"],
                        "tool_calls": len(TRACE), "error": error})
        print("DONE", trial, "PASS" if assessment["passed"] else "FAIL", flush=True)
    summary = {"run_id": run_id, "passed": sum(row["grade"]["passed"] for row in results),
               "protocol_version": 2, "strict_output_passed": sum(row["strict_output_passed"] for row in results),
               "parser_policy": "唯一 JSON 对象可显式恢复，严格格式分数单独保存；不是原始严格测试的重评分。",
               "total": len(results), "results": results, "model": os.environ["DIAG_MODEL"],
               "limits": ["固定开发案例", "单变量预定义修复动作", "微型随机模型两步CPU验证", "没有非Agent多轮反馈对照"]}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pointer = "latest-multistep-smoke.json" if args.trial else "latest-multistep.json"
    (ROOT / "reports" / pointer).write_text(json.dumps({"folder": str(folder), "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "total": len(results), "folder": str(folder)}), flush=True)
    if summary["passed"] != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
