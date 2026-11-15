"""Bounded, receipt-driven CUDA OOM tools for one DeerFlow run."""
from copy import deepcopy
import os
from subprocess import TimeoutExpired
from threading import RLock
from typing import Literal

from langchain_core.tools import tool

from experiment_diagnostics.materials import redact
from run_gpu_oom import compare_trials, run_trial, training_oom_observed

TRACE = []
LOCK = RLock()
MAX_REPAIRS = 2


def record(name, arguments, result):
    TRACE.append({"tool": name, "args": arguments, "result": deepcopy(result)})
    return result


def execute(variant, micro_batch_size=None):
    try:
        return run_trial(os.environ["DIAG_TORCH_PYTHON"], os.environ["DIAG_LOCAL_MODEL"],
                         "local", variant, micro_batch_size)
    except (OSError, RuntimeError, ValueError, TimeoutExpired) as exc:
        return {"status": "worker_error", "error": redact(str(exc)), "execution_started": False}


def model_view(result):
    return {key: value for key, value in result.items() if key not in {"log", "model_profile"}}


@tool
def inspect_gpu_oom() -> dict:
    """Run the original fixed LoRA configuration and return its observed CUDA failure and batch settings."""
    with LOCK:
        observations = [entry["result"] for entry in TRACE if entry["tool"] == "inspect_gpu_oom"]
        result = (dict(observations[0], cache_hit=True, execution_started=False)
                  if observations else execute("baseline"))
        return model_view(record("inspect_gpu_oom", {}, result))


@tool
def verify_gpu_oom_repair(micro_batch_size: Literal[1, 2, 4]) -> dict:
    """Select a smaller micro-batch and run two GPU optimization steps.

    The worker computes the corresponding gradient accumulation to preserve effective batch size 8.
    Each choice can run once, with at most two choices per session. If a larger choice still OOMs,
    reserve the last attempt for micro-batch 1. A failed trial is not a verification.
    The input, checkpoint, precision and CUDA allocator cap stay fixed.
    """
    arguments = {"micro_batch_size": micro_batch_size}
    with LOCK:
        if type(micro_batch_size) is not int or micro_batch_size not in {1, 2, 4}:
            return record("verify_gpu_oom_repair", arguments, {"status": "unsupported_micro_batch", "execution_started": False})
        observations = [entry["result"] for entry in TRACE if entry["tool"] == "inspect_gpu_oom"]
        attempts = [entry for entry in TRACE if entry["tool"] == "verify_gpu_oom_repair"
                    and entry["result"].get("attempted")]
        result = {"status": "inspect_required", "execution_started": False}
        if observations and training_oom_observed(observations[0]):
            if any(entry["result"].get("verified") for entry in attempts):
                result["status"] = "already_verified"
            elif previous := next((entry["result"] for entry in attempts if entry["args"] == arguments), None):
                result = dict(previous, attempted=False, cache_hit=True, execution_started=False)
            elif len(attempts) >= MAX_REPAIRS:
                result["status"] = "budget_exhausted"
            elif (len(attempts) == MAX_REPAIRS - 1 and micro_batch_size != 1
                  and attempts[-1]["args"]["micro_batch_size"] != 1
                  and training_oom_observed(attempts[-1]["result"])):
                result.update(status="reserve_last_attempt", remaining_micro_batch_size=1)
            else:
                result = execute("repair", micro_batch_size)
                checks = compare_trials(observations[0], result, micro_batch_size)
                result.update(attempted=True, verification_checks=checks, verified=all(checks.values()))
        return model_view(record("verify_gpu_oom_repair", arguments, result))
