"""Operator-bound data recipe, cumulative changes and revision-checked CPU execution."""
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
from .recipe_spec import normalize_recipe, canonical_bytes, recipe_hash, apply_change

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parent.parent / "work"
STATE, TRACE = {}, []
LOCK = RLock()
MAX_SEARCHES_PER_REVISION = 2
MAX_SEARCHES_PER_RECIPE = 4


def blocked_by_labels():
    last = STATE.get("last") or {}
    return (STATE.get("allow_label_rebuild") is False and last.get("status") == "failed"
            and last.get("stage") == "forward" and (last.get("label_problem") or {}).get("code") in {
                "label_shape_mismatch", "no_supervised_targets", "label_out_of_range"})


def reference_attempts():
    return [entry for entry in TRACE if entry["tool"] == "search_recipe_reference"
            and entry["result"].get("search_attempted")]


def reference_available():
    with LOCK:
        attempts = reference_attempts()
        revision = STATE.get("revision")
        return (not blocked_by_labels() and len(attempts) < MAX_SEARCHES_PER_RECIPE
                and sum(entry["result"]["revision"] == revision for entry in attempts) < MAX_SEARCHES_PER_REVISION)


def bind_recipe(recipe, *, allow_label_rebuild=True):
    normalized = normalize_recipe(recipe)
    with LOCK:
        STATE.clear()
        STATE.update(current=normalized, initial=copy.deepcopy(normalized), revision=0,
            changes=[], last=None, executions=0, session=uuid.uuid4().hex,
            allow_label_rebuild=allow_label_rebuild is True)
        TRACE.clear()


def execute_current():
    folder = WORK / "recipe-sessions" / STATE["session"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("revision-" + str(STATE["revision"]) + ".json")
    path.write_bytes(canonical_bytes(STATE["current"]))
    process = subprocess.run([os.environ["DIAG_TORCH_PYTHON"], "-I", "-X", "utf8",
        str(Path(__file__).with_name("recipe_worker.py")), "--recipe", str(path)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    if process.returncode not in (0, 1) or not process.stdout.strip():
        raise RuntimeError("Recipe worker infrastructure failure; no experiment result accepted")
    result = json.loads(process.stdout)
    if ((result.get("status") == "passed") != (process.returncode == 0) or
        result.get("recipe_sha256") != recipe_hash(STATE["current"])):
        raise RuntimeError("Recipe hash or worker exit code disagrees with execution evidence")
    result.update(exit_code=process.returncode, execution_started=True, run_id=uuid.uuid4().hex,
        allow_label_rebuild=STATE["allow_label_rebuild"],
        revision=STATE["revision"], recipe=copy.deepcopy(STATE["current"]), changes=copy.deepcopy(STATE["changes"]))
    STATE["executions"] += 1
    STATE["last"] = copy.deepcopy(result)
    return result


def record(tool_name, arguments, result):
    TRACE.append({"tool": tool_name, "args": arguments, "result": copy.deepcopy(result)})
    return result


@tool
def inspect_recipe() -> dict:
    """Inspect the actual caller-selected bounded JSON experiment. Run its fixed CPU recipe if needed.

    Only random tiny Qwen2, numeric token batches, bounded LoRA/SGD settings. No submitted
    source, shell, downloads, model weights or paths. Cached observations are explicitly marked.
    """
    with LOCK:
        if not STATE:
            raise ValueError("No bound recipe")
        result = execute_current() if STATE["last"] is None else dict(copy.deepcopy(STATE["last"]), execution_started=False, cache_hit=True)
        return record("inspect_recipe", {}, result)


@tool
def change_recipe(expected_revision: int,
    change: Literal["set_lora_targets", "pad_batch", "enable_gradients", "rebuild_causal_labels"],
    target_modules: list[str] | None = None) -> dict:
    """Apply one allowlisted configuration change and rerun the SAME cumulative CPU recipe.

    Inspect first and use its current revision. Targets must be observed internal module names.
    rebuild_causal_labels discards the provided labels and uses the input IDs for this causal-LM
    reproduction; it does not prove task-label correctness. Intermediate new errors are failures.
    No code writing/execution from the user; at most seven actual runs; no mutation after success.
    """
    arguments = {"expected_revision": expected_revision, "change": change, "target_modules": target_modules}
    with LOCK:
        if not STATE or STATE["last"] is None:
            return record("change_recipe", arguments, {"status": "inspect_required", "execution_started": False})
        if type(expected_revision) is not int or expected_revision != STATE["revision"]:
            return record("change_recipe", arguments, {"status": "stale_revision", "revision": STATE["revision"], "execution_started": False})
        if STATE["last"]["status"] == "passed":
            return record("change_recipe", arguments, {"status": "already_verified", "execution_started": False,
                "revision": STATE["revision"], "verified_run_id": STATE["last"]["run_id"]})
        if STATE["executions"] >= 7:
            return record("change_recipe", arguments, {"status": "budget_exhausted", "execution_started": False})
        if change == "rebuild_causal_labels" and not STATE["allow_label_rebuild"]:
            return record("change_recipe", arguments, {"status": "label_policy_denied", "execution_started": False,
                "revision": STATE["revision"], "reason": "The operator requires preserving the supplied labels."})
        if blocked_by_labels():
            return record("change_recipe", arguments, {"status": "blocked_by_label_policy", "execution_started": False,
                "revision": STATE["revision"], "label_problem": copy.deepcopy(STATE["last"]["label_problem"])})
        try:
            updated = apply_change(STATE["current"], change, target_modules,
                STATE["last"].get("available_linear_modules", []))
        except (ValueError, TypeError) as exc:
            return record("change_recipe", arguments, {"status": "rejected", "reason": str(exc), "execution_started": False})
        if recipe_hash(updated) == recipe_hash(STATE["current"]):
            return record("change_recipe", arguments, dict(copy.deepcopy(STATE["last"]), execution_started=False, cache_hit=True))
        previous = copy.deepcopy(STATE)
        STATE["current"] = updated
        STATE["revision"] += 1
        STATE["changes"].append(arguments)
        try:
            result = execute_current()
        except Exception:
            STATE.clear()
            STATE.update(previous)
            raise
        return record("change_recipe", arguments, result)


@tool
def search_recipe_reference(query: str, package: Literal["peft", "pytorch", "transformers"]) -> dict:
    """Retrieve exact-version references. Use at most two searches per revision and four per recipe."""
    with LOCK:
        arguments = {"query": query, "package": package}
        if not STATE or STATE["last"] is None:
            return record("search_recipe_reference", arguments, {"status": "inspect_required", "hits": []})
        if not reference_available():
            status = "blocked_by_label_policy" if blocked_by_labels() else "search_budget_exhausted"
            return record("search_recipe_reference", arguments, {"status": status, "hits": [],
                "reason": "Use the existing execution evidence to make a repair or report the outcome."})
        revision = STATE["revision"]
        normalized = " ".join(query.casefold().split())
        for previous in reference_attempts():
            prior_query = " ".join(previous["args"]["query"].casefold().split())
            if previous["result"]["revision"] == revision and previous["args"]["package"] == package and prior_query == normalized:
                return record("search_recipe_reference", arguments, {"status": "search_already_attempted", "hits": [],
                    "revision": revision, "search_attempted": True, "retrieval_executed": False})
        versions = STATE["last"].get("versions", {})
        version = versions.get("torch" if package == "pytorch" else package)
        result = retrieve(query, package=package, top_k=2, versions={package: version} if version else {})
        return record("search_recipe_reference", arguments, dict(result, revision=revision,
            search_attempted=True, retrieval_executed=True))
