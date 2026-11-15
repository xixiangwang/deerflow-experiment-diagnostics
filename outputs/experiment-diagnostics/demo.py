"""Execute only our fixed CPU fixture, then diagnose and verify its repair."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from experiment_diagnostics import inspect_experiment

ROOT = Path(__file__).resolve().parent


def run(batch_size):
    # Fixed source authored for this demo. No submitted code or shell commands run.
    source = "import sys; batch_size=int(sys.argv[1]); print('steps=', 8 // batch_size)"
    result = subprocess.run([sys.executable, "-I", "-c", source, str(batch_size)],
                            capture_output=True, text=True, timeout=10)
    return {"batch_size": batch_size, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    failed = run(0)
    diagnosis = asyncio.run(inspect_experiment(
        {"log": failed["stderr"], "experiment_config": {"batch_size": 0}}, None))
    repaired = run(2)
    assert failed["exit_code"] != 0
    assert any(f["code"] == "invalid_batch_size" for f in diagnosis["findings"])
    assert repaired["exit_code"] == 0 and "steps= 4" in repaired["stdout"]
    report = {"failed_run": failed, "diagnosis": diagnosis, "repaired_run": repaired,
              "verification": "passed", "model_called": False,
              "scope": "固定 CPU 子进程案例；不是安全隔离沙箱，也不是 LLM 端到端测试。"}
    path = ROOT / "reports" / "demo.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: real failure -> evidence -> fixed fixture -> successful rerun")
    print(path)


if __name__ == "__main__":
    main()
