"""Freeze allowlisted scenario source, not operator environments or submitted files."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_run_sources(folder, entrypoint, root=ROOT):
    root = Path(root).resolve()
    destination = Path(folder).resolve() / "provenance"
    destination.mkdir(parents=True, exist_ok=False)
    paths = set(root.glob("*.py"))
    for name in ("experiment_diagnostics", "deerflow_case"):
        paths.update((root / name).rglob("*.py"))
    paths.update((root / "deerflow_case/skills").rglob("SKILL.md"))
    paths.update(path for path in (root / "deerflow-test.yaml", root / "knowledge/manifest.json") if path.is_file())
    files = []
    for source in sorted(paths):
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError("Source must stay within the scenario root")
        name = source.relative_to(root).as_posix()
        data = source.read_bytes()
        target = destination / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    packages = {}
    for name in ("langchain", "langchain-core", "langchain-openai", "langgraph", "pydantic", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    index = Path(os.environ.get("DIAG_KNOWLEDGE_INDEX", root / "knowledge/index.sqlite3"))
    manifest = {"schema_version": 1, "entrypoint": entrypoint,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.system(), "framework_environment": packages,
        "scenario_files": files, "knowledge_index_sha256": digest(index) if index.is_file() else None,
        "knowledge_index_selection": "operator_override" if os.environ.get("DIAG_KNOWLEDGE_INDEX") else "scenario_default",
        "scope": "场景源码快照与索引哈希；不包含进程环境变量、凭据、用户原文件或模型权重。不是完整容器镜像。"}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def check_run_sources(manifest, root=ROOT):
    root = Path(root).resolve()
    changed = []
    for entry in manifest["scenario_files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != entry["sha256"]:
            changed.append(entry["path"])
    index = Path(os.environ.get("DIAG_KNOWLEDGE_INDEX", root / "knowledge/index.sqlite3"))
    if manifest["knowledge_index_sha256"] is not None and (not index.is_file() or digest(index) != manifest["knowledge_index_sha256"]):
        changed.append("knowledge/index.sqlite3")
    return {"unchanged": not changed, "changed_paths": changed}
