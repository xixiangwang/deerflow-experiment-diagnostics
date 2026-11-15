"""Run paired read-only diagnoses on frozen excerpts from public issue reports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse
import uuid

from langchain_openai import ChatOpenAI

from compare_materials import SYSTEM, agent_usage
from experiment_diagnostics.materials import load_materials, redact
from experiment_diagnostics.model_output import material_contract_errors, parse_model_object
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "evaluation_materials/public-issues-v1.json"
REPORTS = ROOT / "reports"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def model_client():
    options = {"temperature": 0, "timeout": 60, "max_retries": 1, "max_tokens": 2200}
    if urlparse(os.environ["DIAG_BASE_URL"]).hostname == "api.deepseek.com":
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(model=os.environ["DIAG_MODEL"], base_url=os.environ["DIAG_BASE_URL"],
                      api_key=os.environ["DIAG_API_KEY"], **options)


def direct_attempt(path, model):
    material = load_materials(path)["materials"]
    response = model.invoke([("system", SYSTEM), ("human", json.dumps(material, ensure_ascii=False))])
    parsed = parse_model_object(response.content)
    answer = parsed.pop("value")
    errors = material_contract_errors(answer)
    tokens = response.usage_metadata or {}
    return {"answer": answer, "raw_answer": response.content, "format": parsed, "error": None,
            "contract_passed": not errors, "contract_errors": errors, "usage": {**tokens, "model_calls_with_usage": bool(tokens)}}


def deerflow_attempt(path):
    command = [sys.executable, str(ROOT / "analyze_materials.py"), str(path), "--no-latest-pointer"]
    process = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", timeout=350)
    if not process.stdout.strip():
        raise RuntimeError("Materials runner returned no receipt. " + redact(process.stderr[-1500:]))
    receipt = json.loads(process.stdout.strip().splitlines()[-1])
    original = Path(receipt["folder"]) / "report.json"
    report = json.loads(original.read_text(encoding="utf-8"))
    if report["sanitized_input"]["materials"] != load_materials(path)["materials"]:
        raise ValueError("Paired material input changed in transit")
    return {"answer": report["answer"], "raw_answer": report["raw_answer"], "format": report["output_format"],
            "error": report["error"], "contract_passed": report["contract_passed"],
            "contract_errors": report["answer_contract_errors"], "usage": agent_usage(report["events"]),
            "report_source": report.get("report_source", "model"), "normalized_fields": report.get("normalized_fields", []),
            "citation_audit": report["citation_audit"], "trace": report["trace"], "events": report["events"],
            "provenance": report["provenance"], "process_exit": process.returncode}


def review_packet(cases, attempts):
    # Keep method, timing and tool trajectory out of the first review view.
    packet, mapping = [], {}
    fields = ("status", "observations", "hypotheses", "recommendations", "verification_plan", "missing_information", "limitations", "validation_status")
    for case in cases:
        selected = sorted((row for row in attempts if row["id"] == case["id"]), key=lambda row: row["review_id"])
        mapping.update({row["review_id"]: {"id": row["id"], "mode": row["mode"]} for row in selected})
        answers = []
        for row in selected:
            value = row["answer"]
            answers.append({"review_id": row["review_id"], "answer": {field: value.get(field) for field in fields}
                            if isinstance(value, dict) else None})
        packet.append({"case": case["id"], "material": case["material"], "rubric": case["rubric"], "answers": answers})
    return packet, mapping


def main():
    raw_manifest = MANIFEST.read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    manifest = json.loads(raw_manifest)
    cases = manifest["cases"]
    if manifest["version"] != 1 or [case["id"] for case in cases] != [f"i{number:02d}" for number in range(1, 9)]:
        raise ValueError("Unexpected public issue manifest")
    run_id = datetime.now(timezone.utc).strftime("pi-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    folder = REPORTS / "public-issues" / run_id
    folder.mkdir(parents=True)
    source = capture_run_sources(folder, Path(__file__).name)
    (folder / "cases.json").write_bytes(raw_manifest)
    rows = []
    model = model_client()
    print("RUN", folder, flush=True)
    for index, case in enumerate(cases):
        path = folder / (case["id"] + "-input.json")
        write_json(path, case["material"])
        order = ("direct", "deerflow") if index % 2 == 0 else ("deerflow", "direct")
        for mode in order:
            print("BEGIN", case["id"], mode, flush=True)
            start = time.perf_counter()
            try:
                row = direct_attempt(path, model) if mode == "direct" else deerflow_attempt(path)
            except Exception as exc:
                row = {"answer": None, "raw_answer": "", "format": {"strict_json": False},
                       "error": redact(type(exc).__name__ + ": " + str(exc)), "contract_passed": False, "usage": {}}
            row.update(id=case["id"], mode=mode, review_id=uuid.uuid4().hex[:10],
                       seconds=round(time.perf_counter() - start, 2), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            write_json(folder / (case["id"] + "-" + mode + ".json"), row)
            rows.append(row)
            print("DONE", case["id"], mode, row["contract_passed"], row["format"].get("strict_json"), row["error"] or "", flush=True)
            if len(rows) == 1 and row["error"]:
                raise SystemExit("First request failed")
    integrity = check_run_sources(source)
    if not integrity["unchanged"] or hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != manifest_hash:
        raise SystemExit("Frozen public issue input or runtime changed")
    packet, mapping = review_packet(cases, rows)
    write_json(folder / "review-packet.json", packet)
    write_json(folder / "review-key.json", mapping)
    summary = {"folder": folder.relative_to(ROOT).as_posix(), "model": os.environ["DIAG_MODEL"], "cases": len(cases),
               "attempts": len(rows), "manifest_sha256": manifest_hash, "source_integrity": integrity,
               "max_model_calls": {"direct": 1, "deerflow": 6}, "executed_submitted_code": False, "gpu_used": False,
               "results": {mode: {"contract_passed": sum(r["contract_passed"] for r in rows if r["mode"] == mode),
                   "strict_json": sum(bool(r["format"].get("strict_json")) for r in rows if r["mode"] == mode),
                   "api_errors": sum(r["error"] is not None for r in rows if r["mode"] == mode),
                   **{key: sum(r["usage"].get(key, 0) for r in rows if r["mode"] == mode)
                      for key in ("input_tokens", "output_tokens", "model_calls_with_usage")}} for mode in ("direct", "deerflow")}}
    write_json(folder / "summary.json", summary)
    write_json(REPORTS / "public-issues-summary.json", summary)
    print(json.dumps(summary["results"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
