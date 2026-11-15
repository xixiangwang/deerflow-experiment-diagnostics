"""Run a frozen developer suite through the actual DeerFlow client and API model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import yaml

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent / "work"
RUN_ID = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
os.environ["DEER_FLOW_PROJECT_ROOT"] = str(ROOT)
os.environ["DEER_FLOW_HOME"] = str(WORK / ("evaluation-state-" + RUN_ID))
os.environ["DEER_FLOW_SKILLS_PATH"] = str(ROOT / "deerflow_case/skills")
os.environ.setdefault("DIAG_KNOWLEDGE_INDEX", str(ROOT / "knowledge/index.sqlite3"))


def final_answer(events):
    chunks = {}
    for event in events:
        data = event["data"]
        if event["type"] == "messages-tuple" and data.get("type") == "ai" and data.get("content"):
            chunks.setdefault(data.get("id", "answer"), []).append(data["content"])
    return "".join(next(reversed(chunks.values()))) if chunks else ""


def grade(case, answer, trace):
    checks = {"structured_output": isinstance(answer, dict), "correct_status": False,
              "inspected_expected_case": False, "case_isolation": True,
              "verified_outcome": False}
    inspections = [call for call in trace if call["tool"] == "inspect_experiment_case"]
    repairs = [call for call in trace if call["tool"] == "verify_experiment_repair"]
    checks["inspected_expected_case"] = any(call["args"]["case_id"] == case["id"] for call in inspections)
    checks["case_isolation"] = all(call["args"].get("case_id", case["id"]) == case["id"] for call in trace)
    if isinstance(answer, dict):
        checks["correct_status"] = answer.get("status") == case["expected_status"] and answer.get("case_id") == case["id"]
    if case["expected_fix"]:
        checks["verified_outcome"] = any(
            call["args"]["case_id"] == case["id"] and call["args"]["fix_type"] == case["expected_fix"]
            and call["result"].get("status") == "passed"
            and call["result"].get("finite_gradients") is True
            and call["result"].get("parameters_changed") is True for call in repairs)
        checks["fault_observed"] = any(call["result"].get("status") == "failed" for call in inspections)
        checks["retrieval_used"] = any(call["tool"] == "retrieve_experiment_evidence" for call in trace)
    else:
        checks["verified_outcome"] = not repairs
        expected_inspection = "passed" if case["expected_status"] == "no_fault_observed" else "insufficient_evidence"
        checks["boundary_observed"] = any(call["result"].get("status") == expected_inspection for call in inspections)
    task_checks = {name: value for name, value in checks.items() if name != "retrieval_used"}
    return {"passed": all(checks.values()), "protocol_passed": all(checks.values()),
            "task_outcome_passed": all(task_checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=[f"exp-{n}" for n in range(101, 107)])
    args = parser.parse_args()
    missing = [key for key in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL") if not os.environ.get(key)]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    config = yaml.safe_load((ROOT / "deerflow-test.yaml").read_text(encoding="utf-8"))
    config["models"] = [{"name": "case-model", "use": "langchain_openai:ChatOpenAI", "model": "$DIAG_MODEL",
                         "api_key": "$DIAG_API_KEY", "base_url": "$DIAG_BASE_URL", "temperature": 0,
                         "max_retries": 1, "timeout": 60, "max_tokens": 1800,
                         "extra_body": {"thinking": {"type": "disabled"}}}]
    config["tools"] = [{"name": name, "group": "diagnosis", "use": "deerflow_case.batch_tools:" + name}
                       for name in ("inspect_experiment_case", "retrieve_experiment_evidence", "verify_experiment_repair")]
    path = WORK / ("evaluation-" + RUN_ID + ".yaml")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    os.environ["DEER_FLOW_CONFIG_PATH"] = str(path)
    from deerflow.client import DeerFlowClient
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from deerflow_case.batch_tools import TRACE
    client = DeerFlowClient(config_path=str(path), model_name="case-model", thinking_enabled=False,
                            subagent_enabled=False, available_skills={"experiment-investigation"},
                            middlewares=[ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")])
    assert client._app_config.get_model_config("case-model").use == "langchain_openai:ChatOpenAI"
    frozen_bytes = (ROOT / "eval_cases.json").read_bytes()
    cases = json.loads(frozen_bytes)["cases"]
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
    folder = ROOT / "reports/evaluations" / RUN_ID
    folder.mkdir(parents=True)
    results = []
    for case in cases:
        print("BEGIN", case["id"], flush=True)
        start, offset = time.perf_counter(), len(TRACE)
        events, error = [], None
        try:
            prompt = ("/experiment-investigation 请检查实验 " + case["id"] +
                      "。先读取该实验的实际证据，再决定是否检索或验证。严格按技能要求输出一个 JSON 对象。")
            for event in client.stream(prompt, thread_id=RUN_ID + "-" + case["id"], recursion_limit=90):
                events.append({"type": event.type, "data": event.data})
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        text = final_answer(events)
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json") and cleaned.endswith("```"):
                cleaned = cleaned[7:-3].strip()
            answer = json.loads(cleaned)
        except ValueError:
            answer = None
        trace = TRACE[offset:]
        result = {"case_id": case["id"], "elapsed_seconds": time.perf_counter() - start,
                  "answer": answer, "raw_final_answer": text, "trace": trace, "events": events,
                  "error": error, "grade": grade(case, answer, trace), "real_llm_used": True,
                  "deerflow_client_used": True, "model": os.environ["DIAG_MODEL"]}
        if error:
            result["grade"].update(passed=False, protocol_passed=False, task_outcome_passed=False)
        (folder / (case["id"] + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (folder / (case["id"] + ".md")).write_text(text, encoding="utf-8")
        results.append({"case_id": case["id"], "grade": result["grade"], "error": error,
                        "tool_calls": len(trace), "elapsed_seconds": result["elapsed_seconds"]})
        print("DONE", case["id"], "PASS" if result["grade"]["passed"] else "FAIL", flush=True)
    summary = {"run_id": RUN_ID, "cases_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
               "evaluation_schema_version": 2,
               "passed": sum(result["grade"]["passed"] for result in results), "total": len(results),
               "task_outcomes_passed": sum(result["grade"]["task_outcome_passed"] for result in results),
               "axes": {"passed": "严格流程：故障案例要求检索及真实验证。",
                        "task_outcomes_passed": "结果核查：要求结论与实际执行一致，但不强制不必要的检索。"},
               "results": results, "model": os.environ["DIAG_MODEL"], "deerflow_version": "2.0.0",
               "scope": "开发回归集，含主动构造的故障和边界案例；标签未放入提示词，不是独立终审集或诊断准确率。"}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports/latest-evaluation.json").write_text(json.dumps({"folder": str(folder), "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "total": len(results), "folder": str(folder)}), flush=True)
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
