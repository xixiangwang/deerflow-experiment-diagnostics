"""One actual official-framework/model-selected repair of the local pretrained GPU fixture."""
import json
import os
from pathlib import Path
import time
import uuid
from experiment_diagnostics.materials import redact
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources
from deerflow_case.runtime import create_api_client, final_model_text

ROOT = Path(__file__).resolve().parent


def main():
    if not os.environ.get("DIAG_TORCH_PYTHON") and not os.environ.get("DIAG_SLURM_PROFILE"):
        raise SystemExit("DIAG_TORCH_PYTHON must select the existing torch/peft environment")
    run_id = "pretrained-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/pretrained" / run_id
    folder.mkdir(parents=True)
    provenance = capture_run_sources(folder, "run_pretrained.py")
    client = create_api_client(run_id, "deerflow_case.pretrained_tools",
        ["inspect_pretrained_trial", "verify_pretrained_repair", "lookup_pretrained_reference"], "pretrained-experiment", 8)
    from deerflow_case.pretrained_tools import TRACE
    events, error = [], None
    try:
        for event in client.stream("/pretrained-experiment 检查当前受控本地 0.5B LoRA 配置错误，并在资源允许时验证修复。",
                                   thread_id=run_id, recursion_limit=100):
            events.append({"type": event.type, "data": event.data})
    except Exception as exc:
        error = redact(type(exc).__name__ + ": " + str(exc))
    text = final_model_text(events)
    parsed = parse_model_object(text)
    answer = parsed["value"]
    inspections = [call["result"] for call in TRACE if call["tool"] == "inspect_pretrained_trial"]
    validations = [call["result"] for call in TRACE if call["tool"] == "verify_pretrained_repair"]
    validation = validations[-1] if validations else {}
    source_integrity = check_run_sources(provenance)
    checks = {"observed_real_error": bool(inspections) and inspections[0].get("status") == "failed"
                  and inspections[0].get("stage") == "adapter_injection" and inspections[0].get("pretrained_weights_used") is True,
              "gpu_validation_passed": validation.get("status") == "passed" and validation.get("device") == "cuda",
              "finite_gradients": validation.get("gradients_finite") is True,
              "adapter_updated": validation.get("parameters_changed") is True,
              "base_frozen": validation.get("base_parameters_frozen") is True,
              "original_weight_stat_unchanged": validation.get("original_weight_stat_unchanged") is True,
              "model_reports_verified": isinstance(answer, dict) and answer.get("status") == "verified_repair",
              "no_runtime_error": error is None, "scenario_source_unchanged": source_integrity["unchanged"]}
    if os.environ.get("DIAG_SLURM_PROFILE"):
        trials = inspections + validations
        checks["slurm_resources_released"] = bool(trials) and all(
            item.get("slurm", {}).get("resources_released") is True for item in trials)
        checks["one_gpu_per_slurm_trial"] = bool(trials) and all(
            "gres/gpu=1" in item.get("slurm", {}).get("allocated_tres", "").split(",") for item in trials)
    result = {"run_id": run_id, "model": os.environ.get("DIAG_MODEL"), "real_llm_used": True,
              "deerflow_version": "2.0.0", "trace": TRACE, "events": events, "answer": answer,
              "execution_backend": "ssh_slurm" if os.environ.get("DIAG_SLURM_PROFILE") else "local_subprocess",
              "raw_answer": text, "format": {k: v for k, v in parsed.items() if k != "value"},
              "checks": checks, "passed": all(checks.values()), "error": error,
              "provenance": {"manifest": "provenance/manifest.json", "source_integrity": source_integrity},
              "limits": ["受控现有 0.5B 模型配置故障", "两条自编短文本、两步 GPU 优化", "不是微调质量或准确率评测",
                         "不是 QLoRA", "权重未调用保存，文件大小/mtime 不变不是全内容哈希证明"]}
    (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "answer.md").write_text(text, encoding="utf-8")
    (ROOT / "reports/latest-pretrained.json").write_text(json.dumps({"folder": str(folder), "passed": result["passed"],
        "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), "passed": result["passed"], "checks": checks}, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
