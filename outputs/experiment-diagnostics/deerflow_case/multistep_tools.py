"""One serial trial session with cumulative patches and revision-checked real reruns."""
import copy
import json
import os
from pathlib import Path
import subprocess
from threading import RLock
from typing import Literal
import uuid
from langchain_core.tools import tool
from experiment_diagnostics.knowledge import retrieve

TRACE = []
STATE = {}
LOCK = RLock()
REPAIRS = {"set_target_modules", "pad_sequences", "enable_grad", "align_causal_labels"}


def bind_trial(trial):
    if trial not in {"trial-201", "trial-202", "trial-203"}:
        raise ValueError("Unknown trial")
    with LOCK:
        STATE.clear()
        STATE.update(trial=trial, revision=0, patches=[], targets=["q_proj", "v_proj"], last=None)
        TRACE.clear()


def execute_current():
    command = [os.environ["DIAG_TORCH_PYTHON"], "-I", str(Path(__file__).with_name("multistep_worker.py")),
               "--trial", STATE["trial"], "--patches", json.dumps(STATE["patches"]),
               "--target-modules", json.dumps(STATE["targets"])]
    process = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    if process.returncode not in (0, 1) or not process.stdout.strip():
        raise RuntimeError("Trial worker infrastructure failed")
    result = json.loads(process.stdout)
    if (result.get("status") == "passed") != (process.returncode == 0):
        raise RuntimeError("Worker status and process exit code disagree")
    result.update(exit_code=process.returncode, execution_started=True, run_id=uuid.uuid4().hex,
                  revision=STATE["revision"], scope="固定微型 CPU 多故障实验，不是任意用户代码或安全隔离沙箱。")
    STATE["last"] = copy.deepcopy(result)
    return result


@tool
def inspect_trial() -> dict:
    """Read the bound trial's current actual run. Executes only the fixed CPU worker if no run exists.

    Returns the current revision, already applied patches, failure stage and observed evidence.
    Cached results are identified, not represented as a new execution. No user source execution.
    """
    with LOCK:
        if not STATE:
            raise ValueError("No bound trial")
        result = execute_current() if STATE["last"] is None else dict(copy.deepcopy(STATE["last"]), execution_started=False, cache_hit=True)
        TRACE.append({"tool": "inspect_trial", "args": {}, "result": copy.deepcopy(result)})
        return result


@tool
def apply_trial_repair(expected_revision: int,
        repair: Literal["set_target_modules", "pad_sequences", "enable_grad", "align_causal_labels"],
        target_modules: list[str] | None = None) -> dict:
    """Apply ONE predefined repair and rerun the complete fixed experiment with all prior patches.

    expected_revision must match the last observation. One change per rerun preserves causal
    attribution; intermediate errors are NOT a successful repair. set_target_modules needs
    actual observed module names. Two finite backward/optimizer steps and changed parameters
    are required for success. No arbitrary code, file paths, commands or GPU training.
    """
    with LOCK:
        if not STATE or STATE["last"] is None:
            raise ValueError("Inspect the trial before applying repairs")
        if type(expected_revision) is not int or expected_revision != STATE["revision"]:
            result = {"status": "stale_revision", "execution_started": False, "revision": STATE["revision"]}
        elif STATE["last"]["status"] == "passed":
            result = {"status": "already_verified", "execution_started": False, "revision": STATE["revision"],
                      "verified_run_id": STATE["last"]["run_id"]}
        elif repair not in REPAIRS:
            raise ValueError("Unsupported repair")
        elif repair in STATE["patches"]:
            result = dict(copy.deepcopy(STATE["last"]), execution_started=False, cache_hit=True)
        else:
            previous = copy.deepcopy(STATE)
            if repair == "set_target_modules":
                allowed = set(STATE["last"].get("available_linear_modules", [])) - {"lm_head"}
                if not target_modules or len(target_modules) > 8 or not set(target_modules) <= allowed:
                    raise ValueError("Use 1..8 observed internal module names")
                STATE["targets"] = list(target_modules)
            STATE["patches"].append(repair)
            STATE["revision"] += 1
            try:
                result = execute_current()
            except Exception:
                STATE.clear()
                STATE.update(previous)
                raise
        TRACE.append({"tool": "apply_trial_repair", "args": {"expected_revision": expected_revision,
                      "repair": repair, "target_modules": target_modules}, "result": copy.deepcopy(result)})
        return result


@tool
def lookup_trial_reference(query: str, package: Literal["peft", "pytorch", "transformers"]) -> dict:
    """Retrieve official version-labeled references; relevance does not prove compatibility or a repair."""
    result = retrieve(query, package=package, top_k=2)
    TRACE.append({"tool": "lookup_trial_reference", "args": {"query": query, "package": package}, "result": result})
    return result
