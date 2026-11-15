"""Same-tools, same-feedback developer comparison: official DeerFlow vs a thin tool loop."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from deerflow_case import multistep_tools as tools
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.materials import redact
from run_multistep import grade_trial

ROOT = Path(__file__).resolve().parent


def lean_loop(trial):
    tools.bind_trial(trial)
    available = {tool.name: tool for tool in (tools.inspect_trial, tools.apply_trial_repair, tools.lookup_trial_reference)}
    model = ChatOpenAI(model=os.environ["DIAG_MODEL"], base_url=os.environ["DIAG_BASE_URL"],
        api_key=os.environ["DIAG_API_KEY"], temperature=0, max_retries=1, timeout=60,
        max_tokens=2200, extra_body={"thinking": {"type": "disabled"}}).bind_tools(list(available.values()))
    skill = (ROOT / "deerflow_case/skills/custom/multistep-experiment/SKILL.md").read_text(encoding="utf-8")
    history = [SystemMessage(content=skill), HumanMessage(content="排查并验证当前绑定实验 " + trial + "，根据实际执行反馈决定下一步。")]
    generations, final, error = [], "", None
    try:
        for _ in range(10):
            response = model.invoke(history)
            generations.append({"content": response.content, "tool_calls": response.tool_calls,
                                "usage": response.usage_metadata or {}})
            history.append(response)
            if not response.tool_calls:
                final = response.content
                break
            for call in response.tool_calls:
                if call["name"] not in available:
                    result = {"status": "not_permitted"}
                else:
                    result = available[call["name"]].invoke(call["args"])
                history.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"], name=call["name"]))
    except Exception as exc:
        error = redact(type(exc).__name__ + ": " + str(exc))
    parsed = parse_model_object(final)
    grade = grade_trial(parsed["value"], tools.TRACE)
    if error:
        grade["passed"] = False
    return {"answer": parsed["value"], "raw_answer": final, "format": {k: v for k, v in parsed.items() if k != "value"},
            "trace": list(tools.TRACE), "grade": grade, "error": error, "generations": generations,
            "model_calls": len(generations), "input_tokens": sum(g["usage"].get("input_tokens", 0) for g in generations),
            "output_tokens": sum(g["usage"].get("output_tokens", 0) for g in generations)}


def official_loop(trial):
    process = subprocess.run([sys.executable, str(ROOT / "run_multistep.py"), "--trial", trial],
                             capture_output=True, encoding="utf-8", errors="replace", timeout=500)
    if not process.stdout.strip():
        raise RuntimeError("Official client failed: " + process.stderr[-1000:])
    receipt = json.loads(process.stdout.strip().splitlines()[-1])
    result = json.loads((Path(receipt["folder"]) / (trial + ".json")).read_text(encoding="utf-8"))
    usage = {}
    for event in result["events"]:
        if event["type"] == "values":
            for message in event["data"].get("messages", []):
                if message.get("type") == "ai" and message.get("usage_metadata"):
                    usage[message["id"]] = message["usage_metadata"]
    result.update(model_calls=len(usage), input_tokens=sum(u.get("input_tokens", 0) for u in usage.values()),
                  output_tokens=sum(u.get("output_tokens", 0) for u in usage.values()), source_folder=receipt["folder"])
    return result


def main():
    missing = [key for key in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL", "DIAG_TORCH_PYTHON") if not os.environ.get(key)]
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))
    run_id = "loop-comparison-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/loop-comparisons" / run_id
    folder.mkdir(parents=True)
    frozen = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "compare_multistep.py", "run_multistep.py", "deerflow_case/multistep_tools.py",
        "deerflow_case/multistep_worker.py", "deerflow_case/runtime.py",
        "deerflow_case/skills/custom/multistep-experiment/SKILL.md")}
    results = []
    for number, trial in enumerate(("trial-201", "trial-202", "trial-203")):
        modes = ["lean", "deerflow"] if number % 2 == 0 else ["deerflow", "lean"]
        pair = {"trial": trial, "attempts": {}}
        for mode in modes:
            print("BEGIN", trial, mode, flush=True)
            start = time.perf_counter()
            try:
                result = lean_loop(trial) if mode == "lean" else official_loop(trial)
            except Exception as exc:
                result = {"grade": {"passed": False}, "error": redact(type(exc).__name__ + ": " + str(exc)), "trace": []}
            result["seconds"] = time.perf_counter() - start
            pair["attempts"][mode] = result
            (folder / (trial + "-" + mode + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print("DONE", trial, mode, "PASS" if result["grade"]["passed"] else "FAIL", flush=True)
        results.append(pair)
    assert frozen == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in frozen}
    summary = {"run_id": run_id, "pipeline_hashes": frozen, "trials": 3,
        "results": {mode: {"evidence_checks_passed": sum(pair["attempts"][mode]["grade"]["passed"] for pair in results),
                           "strict_json": sum(pair["attempts"][mode].get("format", pair["attempts"][mode].get("output_format", {})).get("strict_json", False) for pair in results),
                           "model_calls": sum(pair["attempts"][mode].get("model_calls", 0) for pair in results),
                           "mean_seconds": sum(pair["attempts"][mode]["seconds"] for pair in results) / 3,
                           "input_tokens": sum(pair["attempts"][mode].get("input_tokens", 0) for pair in results),
                           "output_tokens": sum(pair["attempts"][mode].get("output_tokens", 0) for pair in results)}
                    for mode in ("lean", "deerflow")},
        "limits": ["相同工具/技能/反馈/每轮单变量约束，最多10次调用", "轻量循环也是工具调用Agent，不是纯RAG",
                   "官方框架附加系统提示与初始化开销不同", "固定开发案例，非独立泛化评测", "未证明框架质量优势"]}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports/latest-loop-comparison.json").write_text(json.dumps({"folder": str(folder), "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"]), flush=True)


if __name__ == "__main__":
    main()
