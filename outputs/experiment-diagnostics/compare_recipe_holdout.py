"""Paired CPU holdout: one-shot proposal, thin tool loop, and official DeerFlow."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deerflow_case import recipe_tools as tools
from deerflow_case.recipe_holdout import CHANGES, MANIFEST, PREFLIGHT, fingerprint, grade, read_cases
from deerflow_case.recipe_runtime import create_recipe_client, is_worker_report
from deerflow_case.runtime import final_model_text
from experiment_diagnostics.materials import redact
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources

ROOT = Path(__file__).resolve().parent
SKILL = ROOT / "deerflow_case/skills/custom/declarative-experiment/SKILL.md"
MODES = ("direct", "thin", "deerflow")
TOOL_NAMES = ("inspect_recipe", "change_recipe", "search_recipe_reference")
MAX_CALLS = 10

DIRECT_SYSTEM = """你收到受限 tiny Qwen2 CPU 实验的真实初次运行结果（含原配置、模块名、错误）。
只做一次诊断并给出按执行顺序排列的最少修改，不能调用工具、不能声称复跑。
允许的修改是 set_lora_targets（指定观察到的内部投影名，排除 lm_head）、pad_batch（对变长数据 padding）、
enable_gradients（开启梯度）、rebuild_causal_labels（用输入 token 重建标签，仅操作者授权时允许）。
同时检查可能被首个异常遮挡的后续问题，保留有效标签和无需修改的配置。
输出一个 JSON 对象，无代码围栏。字段：status（repair_proposed、baseline_observed 或 unresolved），
changes（按顺序列出对象，每项有 change，只有 set_lora_targets 需要 target_modules 数组），
label_policy_changed（提议重建则 true），blocked_by_label_policy（无法在保留标签的约束下继续则 true），
observations 和 limitations（中文数组）。遇到标签受限的案例，仍可先列出有实际意义的上游必要修改。
"""


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def model_client():
    options = {"temperature": 0, "timeout": 60, "max_retries": 1, "max_tokens": 2200}
    if urlparse(os.environ["DIAG_BASE_URL"]).hostname == "api.deepseek.com":
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(model=os.environ["DIAG_MODEL"], base_url=os.environ["DIAG_BASE_URL"],
                      api_key=os.environ["DIAG_API_KEY"], **options)


def instruction(case):
    label = ("允许在确认标签是运行阻塞时，用输入 token 重建标签，且须明确披露。" if case["allow_label_rebuild"] else
             "操作者要求保留提供的 labels，禁止重建或替换。若因此无法继续，保留证据，报告 unresolved 和 blocked_by_label_policy=true。")
    return ("检查当前绑定的 CPU 实验，只做必要修改。遇到连续错误应继续核对；正常配置保持不变。" + label +
            "最终输出一个 JSON 对象（无围栏），额外包含 blocked_by_label_policy（布尔值）。")


def parse_direct_actions(answer, case):
    if not isinstance(answer, dict) or not isinstance(answer.get("changes"), list) or len(answer["changes"]) > 4:
        raise ValueError("Invalid one-shot change list")
    seen, actions = set(), []
    for item in answer["changes"]:
        if not isinstance(item, dict) or set(item) - {"change", "target_modules"}:
            raise ValueError("Invalid one-shot change object")
        name = item.get("change")
        if not isinstance(name, str) or name not in CHANGES or name in seen:
            raise ValueError("Unknown or duplicate change")
        if name == "rebuild_causal_labels" and not case["allow_label_rebuild"]:
            raise ValueError("Direct proposal violates the label-preservation policy")
        targets = item.get("target_modules")
        if name == "set_lora_targets":
            if not isinstance(targets, list) or not 1 <= len(targets) <= 8 or not all(isinstance(x, str) for x in targets):
                raise ValueError("Missing or invalid target_modules")
        elif targets is not None:
            raise ValueError("target_modules is only allowed for set_lora_targets")
        seen.add(name)
        actions.append({"change": name, "target_modules": targets})
    return actions


def generation(response):
    return {"content": response.content, "tool_calls": response.tool_calls,
            "usage": response.usage_metadata or {}, "finish_reason": response.response_metadata.get("finish_reason")}


def direct(case, audit, generations):
    first = tools.inspect_recipe.invoke({})
    response = model_client().invoke([SystemMessage(content=DIRECT_SYSTEM), HumanMessage(content=instruction(case) +
        "\n初次运行：\n" + json.dumps(first, ensure_ascii=False))])
    generations.append(generation(response))
    parsed = parse_model_object(response.content)
    try:
        actions = parse_direct_actions(parsed["value"], case)
        current = first
        for action in actions:
            current = tools.change_recipe.invoke({"expected_revision": current["revision"], **action})
            if not current.get("execution_started"):
                audit.append("direct replay rejected: " + current["status"])
                break
    except ValueError as exc:
        audit.append(str(exc))
    return response.content


def thin(case, audit, generations):
    available = {name: getattr(tools, name) for name in TOOL_NAMES}
    model = model_client().bind_tools(list(available.values()))
    history = [SystemMessage(content=SKILL.read_text(encoding="utf-8")), HumanMessage(content=instruction(case))]
    for _ in range(MAX_CALLS):
        response = model.invoke(history)
        generations.append(generation(response))
        history.append(response)
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            try:
                result = available[call["name"]].invoke(call["args"])
            except Exception as exc:
                result = {"status": "tool_error", "error": redact(type(exc).__name__ + ": " + str(exc))}
                audit.append(result["error"])
            history.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"], name=call["name"]))
    audit.append("model call limit reached without final report")
    return ""


def official(case, audit, events, identifier):
    client = create_recipe_client(identifier)
    for event in client.stream("/declarative-experiment " + instruction(case), thread_id=identifier, recursion_limit=130):
        events.append({"type": event.type, "data": event.data})
    return final_model_text(events)


def is_budget_notice(message):
    return (not message.get("usage_metadata") and not message.get("tool_calls")
            and isinstance(message.get("content"), str)
            and message["content"].startswith("Model call limits exceeded: run limit ("))


def usage(generations, events):
    if not events:
        values = [gen["usage"] for gen in generations]
        count = len(values)
    else:
        messages = {}
        for event in events:
            if event["type"] == "values":
                for message in event["data"].get("messages", []):
                    if (message.get("type") == "ai" and message.get("id")
                            and not (is_budget_notice(message) or is_worker_report(message))):
                        messages[message["id"]] = message.get("usage_metadata") or {}
        values, count = list(messages.values()), len(messages)
    return {"model_calls": count, "calls_with_token_usage": sum(bool(v) for v in values),
            "input_tokens": sum(v.get("input_tokens", 0) for v in values),
            "output_tokens": sum(v.get("output_tokens", 0) for v in values)}


def attempt(case, mode, versions, identifier):
    tools.bind_recipe(case["recipe"], allow_label_rebuild=case["allow_label_rebuild"])
    audit, generations, events = [], [], []
    error, raw = None, ""
    start = time.perf_counter()
    try:
        if mode == "direct":
            raw = direct(case, audit, generations)
        elif mode == "thin":
            raw = thin(case, audit, generations)
        else:
            raw = official(case, audit, events, identifier)
    except Exception as exc:
        error = redact(type(exc).__name__ + ": " + str(exc))
    parsed = parse_model_object(raw)
    trace = list(tools.TRACE)
    if any(call["result"].get("execution_started") and call["result"].get("versions") != versions for call in trace):
        audit.append("Worker environment differs from frozen oracle")
    verdict = grade(case, mode, parsed["value"], trace, error=error, audit_errors=audit)
    return {"id": case["id"], "mode": mode, "kind": case["kind"], "grade": verdict,
            "raw_answer": raw, "answer": parsed.pop("value"), "format": parsed, "trace": trace,
            "events": events, "generations": generations, "error": error, "audit_errors": audit,
            "usage": usage(generations, events), "seconds": time.perf_counter() - start}


def summarize(rows):
    result = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        result[mode] = {"passed": sum(r["grade"]["passed"] for r in selected),
            "strict_json": sum(r["format"]["strict_json"] for r in selected),
            "passed_and_strict_json": sum(r["grade"]["passed"] and r["format"]["strict_json"] for r in selected),
            "false_completion_claims": sum(r["grade"]["false_completion_claim"] for r in selected),
            "policy_denials": sum(r["grade"]["policy_denial_count"] for r in selected),
            "errors": sum(r["error"] is not None for r in selected),
            "by_kind": {kind: {"passed": sum(r["grade"]["passed"] for r in selected if r["kind"] == kind),
                "total": sum(r["kind"] == kind for r in selected)} for kind in ("repair", "healthy", "blocked")},
            "failed_checks": dict(Counter(key for row in selected for key, ok in row["grade"]["checks"].items() if not ok)),
            "mean_seconds": round(sum(r["seconds"] for r in selected) / max(1, len(selected)), 2),
            "worker_runs": sum(r["grade"]["worker_runs"] for r in selected),
            **{key: sum(r["usage"][key] for r in selected) for key in ("model_calls", "calls_with_token_usage", "input_tokens", "output_tokens")}}
    return result


def main():
    cases = read_cases()
    frozen = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not frozen["passed"] or frozen["frozen_hashes"] != fingerprint():
        raise SystemExit("Run recipe-holdout-preflight before calling the model")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"rh-{stamp}-{uuid.uuid4().hex[:6]}"
    folder = ROOT / "reports/recipe-holdout" / run_id
    folder.mkdir(parents=True)
    source = capture_run_sources(folder, Path(__file__).name)
    (folder / "cases.json").write_bytes(MANIFEST.read_bytes())
    write_json(folder / "run-config.json", {"model": os.environ["DIAG_MODEL"], "case_count": len(cases),
        "frozen_hashes": frozen["frozen_hashes"], "preflight_sha256": hashlib.sha256(PREFLIGHT.read_bytes()).hexdigest(),
        "model_call_limits": {"direct": 1, "thin": MAX_CALLS, "deerflow": MAX_CALLS},
        "max_tokens_per_call": 2200, "worker_runs_limit": 7, "no_case_retries": True,
        "scope": "固定合成留出集；direct 只提交一次建议、评测器回放，不向模型提供后续执行反馈。"})
    print("RUN", folder, flush=True)
    rows = []
    versions = {row["id"]: row["versions"] for row in frozen["cases"]}
    for i, case in enumerate(cases):
        order = MODES[i % 3:] + MODES[:i % 3]
        for mode in order:
            print("BEGIN", case["id"], mode, flush=True)
            row = attempt(case, mode, versions[case["id"]], f"{run_id}-{case['id']}-{mode}")
            write_json(folder / f"{case['id']}-{mode}.json", row)
            rows.append(row)
            print("DONE", case["id"], mode, row["grade"]["passed"], row["format"]["strict_json"], row["error"] or "", flush=True)
            if len(rows) == 1 and row["error"]:
                raise SystemExit("Initial provider request failed; stop rather than multiply failures")
    integrity = check_run_sources(source)
    case_integrity = frozen["frozen_hashes"] == fingerprint()
    summary = {"run_id": run_id, "model": os.environ["DIAG_MODEL"], "cases": len(cases), "attempts": len(rows),
        "by_mode": summarize(rows), "source_integrity": integrity, "case_integrity": case_integrity,
        "folder": folder.relative_to(ROOT).as_posix(), "gpu_used": False,
        "matrix": [{"id": r["id"], "kind": r["kind"], "mode": r["mode"], "passed": r["grade"]["passed"],
                    "strict_json": r["format"]["strict_json"], "checks": r["grade"]["checks"],
                    "actions": r["grade"]["applied_changes"], "error": r["error"]} for r in rows]}
    write_json(folder / "summary.json", summary)
    if not integrity["unchanged"] or not case_integrity:
        raise SystemExit("Frozen source or case integrity check failed")
    write_json(ROOT / "reports/recipe-holdout-summary.json", summary)
    print(json.dumps(summary["by_mode"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
