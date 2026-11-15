"""Real official DeerFlow client + real tools + clearly labeled test-model double."""
import json
import argparse
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
os.environ["DEER_FLOW_PROJECT_ROOT"] = str(ROOT)
os.environ["DEER_FLOW_CONFIG_PATH"] = str(ROOT / "deerflow-test.yaml")
os.environ["DEER_FLOW_HOME"] = str(WORKSPACE / "work" / ("deerflow-run-" + uuid.uuid4().hex[:8]))
os.environ["DEER_FLOW_SKILLS_PATH"] = str(ROOT / "deerflow_case/skills")
os.environ.setdefault("DIAG_KNOWLEDGE_INDEX", str(ROOT / "knowledge/index.sqlite3"))

from deerflow.client import DeerFlowClient
from deerflow_case.tools import TRACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-mode", choices=["test", "local", "api"], default="test")
    args = parser.parse_args()
    config_path = ROOT / "deerflow-test.yaml"
    if args.model_mode in ("local", "api"):
        import yaml
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if args.model_mode == "local":
            config["models"][0]["use"] = "deerflow_case.local_model:LocalChatModel"
            config["models"][0]["model"] = "local-qwen-0.5b"
        else:
            missing = [name for name in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL") if not os.environ.get(name)]
            if missing:
                raise SystemExit("Missing local environment variables: " + ", ".join(missing))
            config["models"][0].update(use="langchain_openai:ChatOpenAI", model="$DIAG_MODEL",
                api_key="$DIAG_API_KEY", base_url="$DIAG_BASE_URL", temperature=0, max_retries=1, timeout=60,
                max_tokens=1800, extra_body={"thinking": {"type": "disabled"}})
        config_path = WORKSPACE / "work" / f"deerflow-{args.model_mode}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    os.environ["DEER_FLOW_CONFIG_PATH"] = str(config_path)
    client = DeerFlowClient(config_path=str(config_path),
                            model_name="integration-test", thinking_enabled=False,
                            subagent_enabled=False, available_skills={"experiment-diagnosis"})
    resolved_use = client._app_config.get_model_config("integration-test").use
    expected_use = {"test": "deerflow_case.test_model:IntegrationTestModel",
                    "local": "deerflow_case.local_model:LocalChatModel",
                    "api": "langchain_openai:ChatOpenAI"}[args.model_mode]
    assert resolved_use == expected_use, (resolved_use, expected_use)
    events = []
    try:
        for event in client.stream("/experiment-diagnosis 诊断 tiny_qwen2_lora 的 target_modules 故障，检索依据并验证修复。",
                                   thread_id="integration-lora", recursion_limit=64):
            events.append({"type": event.type, "data": event.data})
    finally:
        from deerflow_case.test_model import CALLS, SKILL_VISIBLE
        (ROOT / f"reports/deerflow-{args.model_mode}-debug.json").write_text(
            json.dumps({"events": events, "tools": TRACE, "model_inputs": CALLS, "skill_visible": SKILL_VISIBLE},
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if args.model_mode == "local":
            from deerflow_case.local_model import GENERATIONS
            (ROOT / "reports/deerflow-local-attempt.json").write_text(json.dumps({
                "real_llm_used": bool(GENERATIONS), "resolved_model_provider": resolved_use,
                "generations": GENERATIONS, "test_double_calls": len(CALLS), "tools": TRACE,
                "status": "incomplete_until_validated"}, ensure_ascii=False, indent=2), encoding="utf-8")
    expected = ["inspect_lora_experiment", "search_experiment_docs", "validate_lora_fix"]
    actual = [call["tool"] for call in TRACE]
    if args.model_mode == "test":
        assert actual == expected, actual
    elif args.model_mode == "local":
        from deerflow_case.local_model import GENERATIONS
        assert GENERATIONS and not CALLS, "Local mode must actually call the real provider, no test fallback"
        (ROOT / "reports/deerflow-local-attempt.json").write_text(json.dumps({
            "real_llm_used": True, "deerflow_client_used": True, "generations": GENERATIONS,
            "tools": TRACE, "status": "passed" if actual == expected else "incomplete",
            "scope": "Single developer case, not independent quality evaluation"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        assert actual == expected, "Real model did not complete the required tool sequence: " + repr(actual)
    else:
        assert not CALLS, "API mode must not fall back to the test double"
        assert all(name in actual for name in expected), "API model omitted required tools: " + repr(actual)
        assert actual.index(expected[0]) < actual.index(expected[1]) < actual.index(expected[2]), actual
    assert TRACE[0]["result"]["status"] == "failed"
    validation = [call["result"] for call in TRACE if call["tool"] == "validate_lora_fix"][-1]
    assert validation["status"] == "passed"
    if args.model_mode == "test":
        assert any(SKILL_VISIBLE), "Scenario skill was not injected into model context"
    report = {"deerflow_version": "2.0.0", "official_commit": "7e7f0410797693cf882594555ba414e0361d4c6f",
              "deerflow_client_used": True, "real_llm_used": args.model_mode != "test",
              "model_type": {"test": "deterministic_integration_test_double", "local": "local_qwen", "api": "api_model"}[args.model_mode],
              "model_provider": resolved_use,
              "tools": TRACE, "scenario_skill_injected": any(SKILL_VISIBLE) if args.model_mode == "test" else "not_instrumented",
              "events": events, "status": "passed",
              "test_double_calls": len(CALLS),
              "model_name": os.environ.get("DIAG_MODEL") if args.model_mode == "api" else args.model_mode,
              "scope": ("确定性框架集成测试，不评估真实模型推理。" if args.model_mode == "test"
                        else "一条开发案例的真实模型工具闭环，不代表独立诊断准确率或大模型训练效果。")}
    folder = ROOT / "reports"
    stem = "deerflow-integration" if args.model_mode == "test" else "deerflow-" + args.model_mode
    (folder / (stem + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    answers = {}
    for event in events:
        data = event["data"]
        if event["type"] == "messages-tuple" and data.get("type") == "ai" and data.get("content"):
            answers.setdefault(data.get("id", "answer"), []).append(data["content"])
    final_answer = "".join(next(reversed(answers.values()))) if answers else ""
    (folder / (stem + ".md")).write_text(final_answer, encoding="utf-8")
    print("PASS: official DeerFlow -> inspect -> retrieve -> real LoRA validation -> report")
    print("Model:", args.model_mode)


if __name__ == "__main__":
    main()
