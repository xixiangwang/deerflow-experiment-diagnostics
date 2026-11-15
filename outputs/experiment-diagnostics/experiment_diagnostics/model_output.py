"""Recover one unambiguous JSON object without confusing format repair with validation."""
import json


def normalize_material_answer(answer):
    """Preserve a text plan verbatim while delivering the documented array shape."""
    if isinstance(answer, dict) and isinstance(answer.get("verification_plan"), str) and answer["verification_plan"].strip():
        return dict(answer, verification_plan=[answer["verification_plan"]]), ["verification_plan"]
    return answer, []


def material_contract_errors(answer):
    if not isinstance(answer, dict):
        return ["answer must be a JSON object"]
    errors = []
    if answer.get("status") not in {"diagnostic_hypotheses", "insufficient_evidence"}:
        errors.append("unsupported status")
    if answer.get("validation_status") != "not_executed":
        errors.append("read-only analysis cannot claim execution")
    for field in ("observations", "hypotheses", "recommendations", "verification_plan", "missing_information"):
        if not isinstance(answer.get(field), list):
            errors.append(field + " must be an array")
    if not isinstance(answer.get("limitations"), (list, str)):
        errors.append("limitations must be text or an array")
    return errors


def parse_model_object(text):
    if not isinstance(text, str) or len(text) > 100000:
        return {"value": None, "strict_json": False, "recovered": False, "error": "invalid_text"}
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return {"value": value, "strict_json": True, "recovered": False, "error": None}
        return {"value": None, "strict_json": False, "recovered": False, "error": "not_object"}
    except (ValueError, RecursionError):
        pass
    objects, index, attempts = [], 0, 0
    decoder = json.JSONDecoder()
    while index < len(text) and attempts < 500:
        start = text.find("{", index)
        if start < 0:
            break
        attempts += 1
        try:
            value, end = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                objects.append(value)
            index = start + end
        except (ValueError, RecursionError):
            # Never promote a nested object from a broken outer JSON envelope.
            if text[start + 1:].lstrip().startswith('"'):
                return {"value": None, "strict_json": False, "recovered": False, "error": "broken_json_envelope"}
            index = start + 1
    if len(objects) == 1:
        return {"value": objects[0], "strict_json": False, "recovered": True, "error": None}
    return {"value": None, "strict_json": False, "recovered": False,
            "error": "ambiguous_objects" if len(objects) > 1 else "no_object"}
