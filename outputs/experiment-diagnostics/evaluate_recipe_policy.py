"""Verify the recipe entry point on new cases and replay two known regressions."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from compare_recipe_holdout import usage, write_json
from deerflow_case.recipe_holdout import fingerprint, grade, preflight, read_cases
from deerflow_case.recipe_runtime import REPORT_SOURCE
from deerflow_case.recipe_spec import canonical_bytes
from deerflow_case.recipe_tools import MAX_SEARCHES_PER_RECIPE, MAX_SEARCHES_PER_REVISION
from experiment_diagnostics.materials import redact
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources
from run_recipe import FALLBACK_SOURCE

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "evaluation_materials/recipe-policy-v2.json"


def attempt(case, folder):
    path = folder / (case["id"] + "-recipe.json")
    path.write_bytes(canonical_bytes(case["recipe"]))
    command = [sys.executable, str(ROOT / "launch.py"), "recipe", "--input", str(path),
               "--base-url", os.environ["DIAG_BASE_URL"], "--model", os.environ["DIAG_MODEL"]]
    if not case["allow_label_rebuild"]:
        command.append("--preserve-labels")
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    if not process.stdout.strip():
        raise RuntimeError("Recipe process returned no receipt. " + redact(process.stderr[-1600:]))
    receipt = json.loads(process.stdout.strip().splitlines()[-1])
    original = Path(receipt["folder"])
    report = json.loads((original / "report.json").read_text(encoding="utf-8"))
    checks = grade(case, "deerflow", report["answer"], report["trace"], error=report["error"])["checks"]
    executions = [entry["result"] for entry in report["trace"] if entry["result"].get("execution_started")]
    searches = [entry["result"] for entry in report["trace"] if entry["result"].get("search_attempted")]
    revisions = {result["revision"] for result in searches}
    blocked = case["kind"] == "blocked"
    last = executions[-1] if executions else {}
    checks.update(production_entry_passed=process.returncode == 0 and report["grade"]["passed"],
                  search_within_limits=len(searches) <= MAX_SEARCHES_PER_RECIPE and all(
                      sum(s["revision"] == revision for s in searches) <= MAX_SEARCHES_PER_REVISION for revision in revisions),
                  report_source=(report["report_source"] == REPORT_SOURCE if blocked else
                                 report["report_source"] in {"model", FALLBACK_SOURCE}))
    if blocked:
        checks["label_error_matches_worker"] = last.get("label_problem", {}).get("code") == case["label_problem"]
        checks["no_search_after_label_failure"] = not any(s["revision"] == last.get("revision") for s in searches)
        checks["local_report_is_json"] = report["format"]["strict_json"]
    (folder / (case["id"] + "-report.json")).write_bytes((original / "report.json").read_bytes())
    return {"id": case["id"], "kind": case["kind"], "passed": all(checks.values()), "checks": checks,
            "report_source": report["report_source"], "strict_json": report["format"]["strict_json"],
            "delivered_strict_json": report["delivered_strict_json"],
            "stages": [run["stage"] for run in executions], "worker_runs": len(executions),
            "searches": len(searches), "usage": usage([], report["events"]), "error": report["error"],
            "seconds": round(time.perf_counter() - started, 2)}


def verify_case(case, folder):
    print("BEGIN", case["id"], flush=True)
    row = attempt(case, folder)
    write_json(folder / (case["id"] + "-receipt.json"), row)
    print("DONE", case["id"], row["passed"], row["strict_json"], row["usage"]["model_calls"], flush=True)
    return row


def main():
    run_id = datetime.now(timezone.utc).strftime("rp-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/recipe-policy" / run_id
    folder.mkdir(parents=True)
    source = capture_run_sources(folder, Path(__file__).name)
    frozen = fingerprint(MANIFEST)
    (folder / "cases.json").write_bytes(MANIFEST.read_bytes())
    preflight_report = folder / "oracle-preflight.json"
    if not preflight(MANIFEST, preflight_report):
        raise SystemExit("CPU preflight failed")
    cases = read_cases(MANIFEST)
    cases.sort(key=lambda case: case["id"] != "p04")
    rows = []
    replays = {"h10": "no_supervised_targets", "h11": "label_out_of_range"}
    regression_cases = [dict(case, label_problem=replays[case["id"]]) for case in read_cases() if case["id"] in replays]
    print("RUN", folder, flush=True)
    for case in cases + regression_cases:
        row = verify_case(case, folder)
        rows.append(row)
        if len(rows) == 1 and row["error"] and not row["worker_runs"]:
            raise SystemExit("Provider or framework failed before the first worker run")
    summary = {"folder": folder.relative_to(ROOT).as_posix(), "model": os.environ["DIAG_MODEL"],
               "new_cases": [r for r in rows if r["id"].startswith("p")],
               "regression_cases": [r for r in rows if r["id"].startswith("h")],
               "gpu_used": False, "source_integrity": check_run_sources(source),
               "case_integrity": frozen == fingerprint(MANIFEST),
               "preflight_sha256": hashlib.sha256(preflight_report.read_bytes()).hexdigest()}
    write_json(folder / "summary.json", summary)
    write_json(ROOT / "reports/recipe-policy-summary.json", summary)
    return summary["source_integrity"]["unchanged"] and summary["case_integrity"] and all(r["passed"] for r in rows)


if __name__ == "__main__":
    raise SystemExit(not main())
