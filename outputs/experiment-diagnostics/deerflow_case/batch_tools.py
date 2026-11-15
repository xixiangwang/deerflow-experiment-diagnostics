"""Bounded tools for multiple developer cases; expected labels stay outside the prompt."""
import json
import os
from pathlib import Path
import subprocess
from typing import Literal
from langchain_core.tools import tool
from experiment_diagnostics.knowledge import retrieve

TRACE = []
CASE_IDS = {f"exp-{number}" for number in range(101, 107)}


def execute(case_id, mode, fix_type="none", target_modules=None):
    command = [os.environ["DIAG_TORCH_PYTHON"], "-I", str(Path(__file__).with_name("case_worker.py")),
               "--case-id", case_id, "--mode", mode, "--fix-type", fix_type]
    if target_modules is not None:
        command.extend(["--target-modules", json.dumps(target_modules)])
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode not in (0, 1) or not result.stdout.strip():
        raise RuntimeError("Fixture infrastructure failed")
    payload = json.loads(result.stdout)
    payload["exit_code"] = result.returncode
    payload["scope"] = "固定微型 CPU 案例，不是用户真实训练任务，不证明收敛或模型效果。"
    return payload


@tool
def inspect_experiment_case(case_id: str) -> dict:
    """Inspect one allowlisted experiment and run its fixed CPU reproduction when evidence exists.

    Accepts exp-101 through exp-106 only. Reports real configuration, data, versions,
    errors or successful steps. Some cases may lack enough evidence to execute.
    """
    if case_id not in CASE_IDS:
        raise ValueError("Unknown case")
    if case_id == "exp-106":
        result = {"case_id": case_id, "status": "insufficient_evidence", "execution_started": False,
                  "available": {"user_note": "实验中途停止，尚未提供日志和配置。"},
                  "missing": ["error log", "training configuration", "environment versions", "input samples"]}
    else:
        result = execute(case_id, "inspect")
    TRACE.append({"tool": "inspect_experiment_case", "args": {"case_id": case_id}, "result": result})
    return result


@tool
def retrieve_experiment_evidence(query: str, package: Literal["peft", "pytorch", "transformers"]) -> dict:
    """Retrieve official version-labeled evidence. Read the returned version and applicability.

    Topic relevance is not proof of a cause. Compare with observed versions and rerun results.
    """
    result = retrieve(query, top_k=3, package=package)
    TRACE.append({"tool": "retrieve_experiment_evidence", "args": {"query": query, "package": package}, "result": result})
    return result


@tool
def verify_experiment_repair(case_id: str,
        fix_type: Literal["set_target_modules", "pad_sequences", "enable_grad", "restore_label_alignment"],
        target_modules: list[str] | None = None) -> dict:
    """Apply a selected predefined repair to a fixed CPU fixture and perform two real training steps.

    No arbitrary source execution. set_target_modules requires observed projection names.
    pad_sequences pads samples and masks padding labels; enable_grad restores autograd;
    restore_label_alignment restores causal-LM labels to the input token sequence.
    Successful short runs do not prove convergence or task quality.
    """
    if case_id not in CASE_IDS:
        raise ValueError("Unknown case")
    if case_id in {"exp-105", "exp-106"}:
        result = {"status": "not_permitted", "execution_started": False,
                  "reason": "This case has no established executable fault requiring a repair."}
    else:
        allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        if fix_type == "set_target_modules" and (not target_modules or len(target_modules) > 8
                or any(item not in allowed for item in target_modules)):
            raise ValueError("Use 1..8 observed internal linear module names")
        result = execute(case_id, "validate", fix_type, target_modules)
    TRACE.append({"tool": "verify_experiment_repair",
                  "args": {"case_id": case_id, "fix_type": fix_type, "target_modules": target_modules},
                  "result": result})
    return result
