"""Verify an actual CUDA training OOM and a repair with identical effective batches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent


def run_trial(python, model, profile, variant, repair_micro_batch=None):
    worker = ROOT / "deerflow_case/pretrained_worker.py"
    command = [str(python), "-I", "-X", "utf8", str(worker), "--mode", "validate",
               "--model", str(model), "--oom-profile", profile, "--oom-variant", variant]
    if repair_micro_batch is not None:
        command.extend(["--repair-micro-batch", str(repair_micro_batch)])
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"DIAG_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"}}
    process = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=75,
                             env=environment)
    if process.returncode:
        raise RuntimeError(f"{variant} worker exited with {process.returncode}: {process.stderr[-1000:]}")
    return json.loads(process.stdout)


def training_oom_observed(result):
    return (result.get("status") == "failed" and result.get("device") == "cuda"
            and result.get("execution_started") is True and result.get("error_type") == "OutOfMemoryError"
            and result.get("stage") in {"training_forward", "training_backward"}
            and result.get("validation_steps") == 0)


def compare_trials(before, after, repair_micro_batch):
    comparable = ("oom_profile", "gpu", "dtype", "allocator_cap_bytes", "effective_batch_size",
                  "sequence_length", "input_sha256", "config_sha256", "target_modules")
    return {
        "training_oom_observed": training_oom_observed(before),
        "same_inputs_and_budget": all(before.get(key) is not None and before[key] == after.get(key)
                                      for key in comparable),
        "micro_batch_reduced": before.get("micro_batch_size") == 8 and before.get("gradient_accumulation_steps") == 1
        and after.get("micro_batch_size") == repair_micro_batch
        and after.get("gradient_accumulation_steps") == 8 // repair_micro_batch,
        "repair_completed": after.get("status") == "passed" and after.get("validation_steps") == 2,
        "finite_and_updated_adapter": all(after.get(key) is True for key in
                                         ("gradients_finite", "parameters_finite", "parameters_changed", "base_parameters_frozen")),
        "original_weight_stat_unchanged": all(trial.get("original_weight_stat_unchanged") is True
                                              for trial in (before, after)),
    }


def run_pair(python, model, profile):
    trials = {variant: run_trial(python, model, profile, variant) for variant in ("baseline", "repair")}
    checks = compare_trials(trials["baseline"], trials["repair"], 1)
    return {"profile": profile, "trials": trials, "checks": checks, "passed": all(checks.values()),
            "repair_selected_by": "fixed_experiment", "real_llm_used": False,
            "scope": "CUDA 分配器进程上限下的显存复现和短程 LoRA 复验。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-profile", choices=["a100"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.worker_profile) != bool(args.worker_model):
        parser.error("Both worker arguments are required together")
    if args.worker_profile:
        print(json.dumps(run_pair(sys.executable, args.worker_model, args.worker_profile), ensure_ascii=True))
        return

    if profile := os.environ.get("DIAG_SLURM_PROFILE"):
        from deerflow_case.slurm_pretrained import SlurmPretrainedExecutor
        report = SlurmPretrainedExecutor(profile).execute("oom-pair")
        receipt = report["slurm"]
        resources = receipt["allocated_tres"].split(",")
        report["checks"]["slurm_released_one_gpu"] = (receipt["resources_released"] is True and
                                                      "gres/gpu=1" in resources)
        report["passed"] = all(report["checks"].values())
    else:
        report = run_pair(os.environ["DIAG_TORCH_PYTHON"], os.environ["DIAG_LOCAL_MODEL"], "local")
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    path = ROOT / "reports/gpu-oom" / run_id / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "passed": report["passed"], "checks": report["checks"]},
                     ensure_ascii=False))
    raise SystemExit(not report["passed"])


if __name__ == "__main__":
    main()
