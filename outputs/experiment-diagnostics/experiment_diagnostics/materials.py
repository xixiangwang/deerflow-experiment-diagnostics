"""Read selected UTF-8 experiment files as data. No imports/execution of submitted code."""
import ast
import hashlib
import json
from pathlib import Path
import re

MAX_BYTES = 300000
SECRET_NAME = re.compile(r"^(?:api[_-]?key|secret|password|passwd|token|access[_-]?token|auth[_-]?token|authorization|private_key|.*[_-](?:api_key|token|password|secret|secret_access_key))$", re.I)


def redact(value, depth=0):
    if depth > 12:
        raise ValueError("Input nesting exceeds 12 levels")
    if isinstance(value, dict):
        return {key: "[REDACTED]" if SECRET_NAME.fullmatch(key) else redact(item, depth + 1)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, depth + 1) for item in value]
    if isinstance(value, str):
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
        value = re.sub(r"(?i)\bBearer[ \t]+[A-Za-z0-9._-]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?im)(\b(?:api_key|password|access_token|secret)\s*[:=]\s*)[^\r\n]+", r"\1[REDACTED]", value)
    return value


def read_selected(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Input must be a regular file, not a symbolic link")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Selected file exceeds 300000 bytes")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Selected file exceeds 300000 bytes")
    return raw.decode("utf-8-sig"), hashlib.sha256(raw).hexdigest()


def normalize(data):
    if not isinstance(data, dict) or set(data) - {"log", "experiment_config", "environment", "code", "question"}:
        raise ValueError("Unknown input fields")
    for key, limit in (("log", 100000), ("code", 50000), ("question", 3000)):
        if not isinstance(data.get(key, ""), str) or len(data.get(key, "")) > limit:
            raise ValueError("Invalid or oversized " + key)
    for key in ("experiment_config", "environment"):
        if not isinstance(data.get(key, {}), dict):
            raise ValueError(key + " must be an object")
    versions = data.get("environment", {}).get("versions", {})
    if not isinstance(versions, dict) or any(not isinstance(v, str) for v in versions.values()):
        raise ValueError("environment.versions must map names to strings")
    normalized = {"log": "", "code": "", "question": "分析实验问题，给出证据、修改建议和验证计划。",
                  "experiment_config": {}, "environment": {}, **data}
    if len(json.dumps(normalized, allow_nan=False).encode()) > MAX_BYTES:
        raise ValueError("Combined material is too large")
    return redact(normalized)


def load_materials(path):
    path = Path(path)
    sources = []
    if path.is_symlink():
        raise ValueError("Input cannot be a symbolic link")
    if path.is_dir():
        data = {}
        for name, field in (("log.txt", "log"), ("config.json", "experiment_config"),
                            ("environment.json", "environment"), ("code.py", "code")):
            source = path / name
            if not source.exists() and not source.is_symlink():
                continue
            text, digest = read_selected(source)
            data[field] = json.loads(text) if name.endswith(".json") else text
            sources.append({"file": name, "sha256": digest})
        if not sources:
            raise ValueError("Folder has no supported files")
    else:
        text, digest = read_selected(path)
        data = json.loads(text)
        sources.append({"file": path.name, "sha256": digest})
    materials = normalize(data)
    return {"materials": materials, "sources": sources, "submitted_code_executed": False,
            "redactions_applied": redact(data) != data,
            "notice": "常见密钥格式已脱敏，但不是完备的秘密检测器；只发送明确选择的材料。"}


def inspect_materials(bundle):
    from .diagnose import diagnose
    data = bundle["materials"]
    findings = diagnose({"log": data["log"], "config": data["experiment_config"], "environment": data["environment"]})
    syntax = {"provided": bool(data["code"]), "executed": False}
    if data["code"]:
        try:
            tree = ast.parse(data["code"])
            syntax["parsed"] = True
            syntax["calls"] = [{"source": "code", "line": node.lineno, "text": ast.unparse(node.func)}
                               for node in ast.walk(tree) if isinstance(node, ast.Call)][:80]
        except (SyntaxError, RecursionError) as exc:
            syntax.update(parsed=False, error=str(exc), line=getattr(exc, "lineno", None))
    if bundle.get("redactions_applied"):
        syntax["notice"] = "静态分析基于脱敏文本；不能把脱敏引入的语法变化认定为原始代码错误。"
    return {**findings, "syntax": syntax, "validation_executed": False,
            "materials": data, "sources": bundle["sources"],
            "log_lines": [{"source": "log", "line": n, "text": line}
                          for n, line in enumerate(data["log"].splitlines(), 1)],
            "missing": [key for key in ("log", "experiment_config", "environment") if not data[key]]}


def has_minimal_material(bundle):
    data = bundle["materials"]
    return (not any(data[key] for key in ("experiment_config", "environment", "code"))
            and sum(bool(line.strip()) for line in data["log"].splitlines()) <= 1)


def minimal_material_report(inspection):
    # Some single-line errors already contain a concrete symptom the rules can identify.
    if any(item["certainty"] == "observed" for item in inspection["findings"]):
        return None
    lines = [line for line in inspection["log_lines"] if line["text"].strip()]
    observations = ([f'日志第 {lines[0]["line"]} 行报告 {lines[0]["text"].strip()[:600]}'] if lines else [])
    missing = ["完整调用栈和首次报错的位置。", "触发异常的最小代码、实际输入及生效配置。",
               "相关库版本和设备环境。"]
    if lines and "unscale" in lines[0]["text"].casefold():
        missing.append("参数与梯度的实际 dtype，以及生效的 AMP 和梯度缩放设置。")
    return {"status": "insufficient_evidence", "observations": observations, "hypotheses": [],
            "recommendations": [], "verification_plan": ["先用完整调用栈定位失败操作，再核对该操作的实际输入。"],
            "missing_information": missing, "validation_status": "not_executed", "references": [],
            "limitations": ["依据提交材料判断当前证据范围。"]}


def render_material_report(answer):
    sections = {"status": "诊断状态", "observations": "观察证据", "hypotheses": "原因假设",
                "recommendations": "修改建议", "verification_plan": "待执行验证计划",
                "missing_information": "还缺少的信息", "limitations": "限制"}
    sections["references"] = "声明的文档引用（需另核对来源与语义）"
    lines = ["# 实验材料诊断报告", "", "**工具侧事实：本次没有执行提交的代码，没有应用修复，验证未执行。**", ""]
    if not isinstance(answer, dict):
        return "\n".join(lines + ["模型未返回符合要求的结构化答案，请查看原始运行记录。"])
    for key, label in sections.items():
        lines.extend(["## " + label, ""])
        value = answer.get(key, "未提供")
        if isinstance(value, list):
            lines.extend("- " + str(item) for item in value)
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)
