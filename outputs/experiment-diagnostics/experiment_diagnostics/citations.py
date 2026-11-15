"""Check citation provenance only. Matching a retrieved URL does not entail a claim."""


def audit_material_citations(answer, trace):
    available = {}
    for entry in trace:
        if entry.get("tool") == "search_material_references":
            for hit in entry.get("result", {}).get("hits", []):
                available[hit["citation"]] = hit
    errors, verified, seen = [], [], set()
    references = answer.get("references") if isinstance(answer, dict) else None
    if not isinstance(references, list):
        errors.append("references must be an array, including when empty")
        references = []
    if len(references) > 20:
        errors.append("at most 20 references are allowed")
        references = references[:20]
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"citation", "used_for"}:
            errors.append("each reference needs citation and used_for only")
            continue
        citation, purpose = reference["citation"], reference["used_for"]
        if not isinstance(citation, str) or not isinstance(purpose, str) or not 1 <= len(purpose.strip()) <= 2000:
            errors.append("invalid citation or used_for")
            continue
        if citation in seen:
            errors.append("duplicate citation")
            continue
        seen.add(citation)
        hit = available.get(citation)
        if not hit:
            errors.append("citation was not returned by this run's search tools: " + citation[:300])
            continue
        verified.append({"citation": citation, "used_for": purpose, "source_id": hit["id"],
            "source_sha256": hit["source_sha256"], "source_line_start": hit["source_line_start"],
            "source_line_end": hit["source_line_end"], "package": hit["package"], "version": hit["version"],
            "applicability": hit["applicability"], "provenance_matched": True, "claim_entailment_checked": False})
    return {"passed": not errors, "errors": errors, "returned_unique_chunks": len(available),
        "declared_references": len(references), "matched_references": verified,
        "unverified_version_citations": [item["citation"] for item in verified if item["applicability"] != "exact_version"],
        "claim_entailment_checked": False, "citation_completeness_checked": False,
        "scope": "只核对声明引用是否在本次真实检索结果中，保留来源哈希、版本与行号；不判断语义蕴含、引用完整性或兼容性。"}
