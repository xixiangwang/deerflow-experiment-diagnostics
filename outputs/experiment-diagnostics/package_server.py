"""Bundle only reviewed runtime files; never include venvs, models or credentials."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
UPSTREAM = OUTPUTS / "deer-flow"


def main():
    files = []
    for name in ("experiment_diagnostics", "knowledge", "deploy"):
        files.extend(p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.extend(ROOT / name for name in ("server.py", "README.md", "STATUS.md", "pyproject.toml"))
    contract = UPSTREAM / "backend/packages/extension-api"
    files.extend(p for p in (contract / "deerflow_extension_api").rglob("*.py") if "__pycache__" not in p.parts)
    files.extend([contract / "pyproject.toml", UPSTREAM / "LICENSE", OUTPUTS / ".dockerignore"])
    files = sorted(set(files))
    entries = []
    destination = OUTPUTS / "diagnostic-server-bundle.tar.gz"
    with tarfile.open(destination, "w:gz") as archive:
        for file in files:
            assert file.resolve().is_relative_to(OUTPUTS.resolve()) and not file.is_symlink()
            relative = file.relative_to(OUTPUTS).as_posix()
            archive.add(file, arcname=relative, recursive=False)
            entries.append({"path": relative, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    report = {"archive": destination.name, "bytes": destination.stat().st_size,
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "files": entries,
              "scope": "离线 API 运行包，非完整 DeerFlow；无模型、虚拟环境、密钥。"}
    (ROOT / "reports/bundle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    print("Files:", len(files), "Bytes:", destination.stat().st_size)


if __name__ == "__main__":
    main()
