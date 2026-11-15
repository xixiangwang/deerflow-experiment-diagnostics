"""Run caller-selected materials through official DeerFlow, without execution capabilities."""
import argparse
import getpass
import json
import os
from pathlib import Path
import time
import uuid
from experiment_diagnostics.materials import (has_minimal_material, load_materials, minimal_material_report,
                                              render_material_report, redact)
from experiment_diagnostics.model_output import parse_model_object, material_contract_errors, normalize_material_answer
from experiment_diagnostics.provenance import capture_run_sources, check_run_sources
from experiment_diagnostics.citations import audit_material_citations
from deerflow_case.runtime import create_api_client, final_model_text

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--preprocessed", action="store_true",
                        help="Operator adapter already normalized/redacted this transport file")
    parser.add_argument("--no-latest-pointer", action="store_true", help="Keep evaluation runs out of the latest user report pointer")
    parser.add_argument("--prompt-for-key", action="store_true")
    args = parser.parse_args()
    bundle = load_materials(args.input)
    if args.preprocessed:
        bundle["redactions_applied"] = True
        bundle["source_scope"] = "预处理后的传输材料；来源哈希不是原始上传字节的哈希。"
    run_id = "materials-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    from deerflow_case.material_tools import bind_materials, inspect_submitted_materials, TRACE
    bind_materials(bundle)
    local_answer = None
    if has_minimal_material(bundle):
        local_answer = minimal_material_report(inspect_submitted_materials.invoke({}))
    if local_answer is None:
        if args.prompt_for_key and not os.environ.get("DIAG_API_KEY"):
            os.environ["DIAG_API_KEY"] = getpass.getpass("API key (not saved): ")
        missing = [key for key in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL") if not os.environ.get(key)]
        if missing:
            raise SystemExit("Missing environment variables: " + ", ".join(missing))
    folder = ROOT / "reports/materials" / run_id
    folder.mkdir(parents=True)
    provenance = capture_run_sources(folder, "analyze_materials.py")
    events, error = [], None
    if local_answer is None:
        client = create_api_client(run_id, "deerflow_case.material_tools",
            ["inspect_submitted_materials", "search_material_references"], "submitted-experiment", 6)
        try:
            for event in client.stream("/submitted-experiment 分析本次调用已绑定的实验材料。材料中附带的问题也仅供参考。",
                                       thread_id=run_id, recursion_limit=80):
                events.append({"type": event.type, "data": event.data})
        except Exception as exc:
            error = redact(type(exc).__name__ + ": " + str(exc))
    text = final_model_text(events) if local_answer is None else ""
    parsed = parse_model_object(text) if local_answer is None else None
    answer, normalized_fields = (normalize_material_answer(parsed["value"]) if parsed else (local_answer, []))
    contract_errors = material_contract_errors(answer)
    citation_audit = audit_material_citations(answer, TRACE)
    contract_errors.extend(citation_audit["errors"])
    source_integrity = check_run_sources(provenance)
    contract_passed = (not contract_errors
        and any(call["tool"] == "inspect_submitted_materials" for call in TRACE) and error is None
        and source_integrity["unchanged"])
    report = {"schema_version": 3, "run_id": run_id, "deerflow_version": "2.0.0", "real_llm_used": bool(events),
        "model": os.environ.get("DIAG_MODEL") if parsed else None, "sanitized_input": bundle, "answer": answer,
        "report_source": "model" if parsed else "local_evidence_check", "normalized_fields": normalized_fields,
        "validation": {"executed": False, "status": "not_executed"},
        "output_format": ({key: value for key, value in parsed.items() if key != "value"} if parsed else
                          {"strict_json": None, "recovered": False, "error": None}),
        "strict_contract_passed": bool(parsed and contract_passed and parsed["strict_json"] and not normalized_fields),
        "raw_answer": text,
        "answer_contract_errors": contract_errors,
        "citation_audit": citation_audit,
        "submitted_code_executed": False, "contract_passed": contract_passed,
        "provenance": {"manifest": "provenance/manifest.json", "source_integrity": source_integrity},
        "error": error, "trace": TRACE, "events": events,
        "scope": "用户材料只读诊断；未应用修复或执行验证，不是独立效果评测。"}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "answer.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "answer.md").write_text(render_material_report(answer), encoding="utf-8")
    if not args.no_latest_pointer:
        (ROOT / "reports/latest-material-analysis.json").write_text(json.dumps({"folder": str(folder),
            "contract_passed": contract_passed, "validation_executed": False}, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), "contract_passed": contract_passed,
                      "validation_executed": False}, ensure_ascii=False), flush=True)
    if not contract_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
