"""Let DeerFlow select and verify a bounded CUDA OOM repair."""
import json
import os
from pathlib import Path
import time
import uuid

from deerflow_case.gpu_oom_tools import TRACE
from deerflow_case.runtime import create_api_client, final_model_text
from experiment_diagnostics.materials import redact
from experiment_diagnostics.model_output import parse_model_object
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources
from run_gpu_oom import training_oom_observed

ROOT = Path(__file__).resolve().parent


def main():
    if os.environ.get("DIAG_SLURM_PROFILE"):
        raise SystemExit("The GPU OOM Agent uses a local CUDA worker")
    TRACE.clear()
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/gpu-oom-agent" / run_id
    folder.mkdir(parents=True)
    manifest = capture_run_sources(folder, "run_gpu_oom_agent.py")
    events, error = [], None
    try:
        client = create_api_client(run_id, "deerflow_case.gpu_oom_tools",
                                   ["inspect_gpu_oom", "verify_gpu_oom_repair"], "gpu-memory-experiment", 5)
        for event in client.stream("/gpu-memory-experiment 检查这次 LoRA 训练的故障。根据回执选择受限的修复并实际验证，最后给出 JSON。",
                                   thread_id=run_id, recursion_limit=50):
            events.append({"type": event.type, "data": event.data})
    except Exception as exc:
        error = redact(f"{type(exc).__name__}: {exc}")

    raw = final_model_text(events)
    parsed = parse_model_object(raw)
    answer = parsed["value"]
    inspections = [entry["result"] for entry in TRACE if entry["tool"] == "inspect_gpu_oom"]
    verified = next((entry for entry in TRACE if entry["tool"] == "verify_gpu_oom_repair"
                     and entry["result"].get("attempted") and entry["result"].get("verified")), None)
    selected = verified["args"]["micro_batch_size"] if verified else None
    checks = {
        "real_training_oom": bool(inspections) and training_oom_observed(inspections[0]),
        "agent_selected_verified_repair": verified is not None,
        "final_answer_matches_receipt": isinstance(answer, dict) and verified is not None
        and answer.get("status") == "verified_repair" and type(answer.get("selected_micro_batch_size")) is int
        and answer["selected_micro_batch_size"] == selected and type(answer.get("validation_steps")) is int
        and answer["validation_steps"] == verified["result"]["validation_steps"],
        "no_runtime_error": error is None,
        "source_integrity": check_run_sources(manifest)["unchanged"],
    }
    report = {"run_id": run_id, "framework": "DeerFlow v2.0.0", "model": os.environ.get("DIAG_MODEL"),
              "real_llm_used": bool(TRACE or raw), "selected_micro_batch_size": selected,
              "execution_backend": "local_subprocess", "checks": checks, "passed": all(checks.values()),
              "answer": answer, "raw_answer": raw, "format": {k: v for k, v in parsed.items() if k != "value"},
              "trace": TRACE, "events": events, "error": error,
              "scope": "一次受控 CUDA 分配器上限实验，使用合成序列和短程 LoRA 复跑。"}
    path = folder / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(path), "passed": report["passed"], "checks": checks, "error": error},
                     ensure_ascii=False))
    raise SystemExit(not report["passed"])


if __name__ == "__main__":
    main()
