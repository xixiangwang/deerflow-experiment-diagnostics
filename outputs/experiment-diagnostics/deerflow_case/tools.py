"""Config-registered DeerFlow tools with bounded, fixed CPU validation."""
import json
import os
from pathlib import Path
import subprocess
from langchain_core.tools import tool
from experiment_diagnostics.knowledge import retrieve

ROOT = Path(__file__).resolve().parent.parent
TRACE = []


def worker(mode, targets=None):
    executable = Path(os.environ["DIAG_TORCH_PYTHON"])
    command = [str(executable), "-I", str(Path(__file__).with_name("worker.py")), "--mode", mode]
    if targets is not None:
        command.extend(["--target-modules", json.dumps(targets)])
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace",
                            timeout=60, cwd=ROOT)
    payload = json.loads(result.stdout)
    payload["exit_code"] = result.returncode
    if (mode == "inspect" and result.returncode != 1) or (mode == "validate" and result.returncode != 0):
        raise RuntimeError("Fixture did not produce the expected execution status")
    return payload


@tool
def inspect_lora_experiment(case_id: str) -> dict:
    """Run the fixed tiny-Qwen2 CPU LoRA failure and list actual available linear modules.

    Only case_id='tiny_qwen2_lora' is allowed. Executes no user-supplied source.
    """
    if case_id != "tiny_qwen2_lora":
        raise ValueError("Unknown fixture")
    result = worker("inspect")
    TRACE.append({"tool": "inspect_lora_experiment", "args": {"case_id": case_id}, "result": result})
    return result


@tool
def search_experiment_docs(query: str) -> dict:
    """Search official PEFT documents and return exact source lines and version labels.

    Current indexed PEFT docs are version 0.13.2; applicability to installed versions
    is NOT assumed. Runtime inspection and rerun are needed for the actual environment.
    """
    result = retrieve(query, top_k=2, package="peft")
    TRACE.append({"tool": "search_experiment_docs", "args": {"query": query}, "result": result})
    return result


@tool
def validate_lora_fix(case_id: str, target_modules: list[str]) -> dict:
    """Verify a proposed LoRA target-module fix on the fixed tiny CPU fixture.

    Uses a fresh tiny random Qwen2 model and two real backward/optimizer steps;
    checks finite gradients and changed adapter parameters. No arbitrary code execution.
    """
    if case_id != "tiny_qwen2_lora" or not isinstance(target_modules, list) or not 1 <= len(target_modules) <= 8:
        raise ValueError("Invalid fixture or module count")
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if any(not isinstance(item, str) or item not in allowed for item in target_modules):
        raise ValueError("Unsupported target module")
    result = worker("validate", target_modules)
    TRACE.append({"tool": "validate_lora_fix", "args": {"case_id": case_id, "target_modules": target_modules},
                  "result": result})
    return result
