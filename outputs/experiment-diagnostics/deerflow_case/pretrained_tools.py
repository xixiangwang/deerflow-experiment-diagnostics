"""Allowlisted existing-model GPU trial; model path is operator-owned, never a tool argument."""
import json
import copy
import os
from pathlib import Path
import subprocess
from threading import RLock
from langchain_core.tools import tool
from experiment_diagnostics.knowledge import retrieve

TRACE = []
LOCK = RLock()


def execute(mode, targets=None):
    if os.environ.get("DIAG_SLURM_PROFILE"):
        from .slurm_pretrained import SlurmPretrainedExecutor
        return SlurmPretrainedExecutor(os.environ["DIAG_SLURM_PROFILE"]).execute(mode, targets)
    command = [os.environ["DIAG_TORCH_PYTHON"], "-I", "-X", "utf8", str(Path(__file__).with_name("pretrained_worker.py")),
               "--mode", mode, "--model", os.environ["DIAG_LOCAL_MODEL"]]
    if targets is not None:
        command.extend(["--target-modules", json.dumps(targets)])
    process = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    if process.returncode or not process.stdout.strip():
        raise RuntimeError("Pretrained worker did not return a result")
    result = json.loads(process.stdout)
    result["execution_started"] = result.get("execution_started", result.get("status") != "skipped")
    return result


@tool
def inspect_pretrained_trial() -> dict:
    """Run the fixed bad-target LoRA configuration on an existing local Qwen2.5-0.5B GPU model.

    No download or weight saving. Skips if free VRAM is below 1.9GB. This only inspects the
    fixed developer trial and does not run submitted code or arbitrary model paths.
    """
    with LOCK:
        previous = [item["result"] for item in TRACE if item["tool"] == "inspect_pretrained_trial"]
        result = (dict(copy.deepcopy(previous[-1]), execution_started=False, cache_hit=True)
                  if previous else execute("inspect"))
        TRACE.append({"tool": "inspect_pretrained_trial", "args": {}, "result": copy.deepcopy(result)})
        return result


@tool
def verify_pretrained_repair(target_modules: list[str]) -> dict:
    """Validate observed LoRA target names using TWO GPU training steps on the fixed local 0.5B model.

    Uses two authored short text samples. Checks finite gradients and changed adapter weights;
    never saves model weights, downloads a model, runs submitted code, or proves task quality.
    """
    with LOCK:
        observed = [item["result"] for item in TRACE if item["tool"] == "inspect_pretrained_trial"]
        if not observed:
            raise ValueError("Inspect the actual model first")
        attempts = [item for item in TRACE if item["tool"] == "verify_pretrained_repair"]
        if len(attempts) >= 2:
            return {"status": "budget_exhausted", "execution_started": False}
        allowed = set(observed[-1].get("available_linear_modules", [])) - {"lm_head"}
        if not target_modules or len(target_modules) > 8 or not set(target_modules) <= allowed:
            raise ValueError("Choose 1..8 actual observed internal module names")
        result = execute("validate", target_modules)
        TRACE.append({"tool": "verify_pretrained_repair", "args": {"target_modules": target_modules}, "result": copy.deepcopy(result)})
        return result


@tool
def lookup_pretrained_reference(query: str) -> dict:
    """Retrieve PEFT reference passages with version labels; runtime compatibility is not assumed."""
    result = retrieve(query, package="peft", top_k=2)
    TRACE.append({"tool": "lookup_pretrained_reference", "args": {"query": query}, "result": result})
    return result
