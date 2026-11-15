"""Use the training interpreter for metadata-only preflight, without loading model weights."""
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    process = subprocess.run([os.environ["DIAG_TORCH_PYTHON"], "-I", "-X", "utf8", str(root / "deerflow_case/pretrained_worker.py"),
        "--mode", "preflight", "--model", os.environ["DIAG_LOCAL_MODEL"]],
        capture_output=True, encoding="utf-8", errors="replace", timeout=45)
    if process.returncode or not process.stdout.strip():
        raise RuntimeError("Preflight interpreter did not return a report")
    report = json.loads(process.stdout)
    (root / "reports/pretrained-preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report.get("status") != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
