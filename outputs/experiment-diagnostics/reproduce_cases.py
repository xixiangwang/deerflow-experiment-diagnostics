"""Run fixed authored CPU fixtures. Never execute user-supplied source."""
import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from experiment_diagnostics import inspect_experiment

ROOT = Path(__file__).resolve().parent
FIXTURES = [
    {"id": "zero_batch", "expected": "invalid_batch_size", "config": {"batch_size": 0},
     "broken": "batch=0; print(8//batch)", "fixed": "batch=2; print(8//batch)"},
    {"id": "missing_module", "expected": "missing_module", "config": {},
     "broken": "import experiment_fixture_nonexistent_package_91c8",
     "fixed": "import json; print(json.dumps({'steps':4}))"},
    {"id": "missing_data", "expected": "missing_file", "config": {},
     "broken": "from pathlib import Path; Path('missing_training_data.json').read_text()",
     "fixed": "import json; from pathlib import Path; Path('train.json').write_text('[1,2]'); print(json.loads(Path('train.json').read_text()))"},
    {"id": "nonfinite_loss", "expected": "nonfinite_loss", "config": {},
     "broken": "import math; loss=float('nan'); print('loss=',loss); assert math.isfinite(loss)",
     "fixed": "import math; loss=0.5; print('loss=',loss); assert math.isfinite(loss)"},
    {"id": "target_range", "expected": "target_out_of_bounds", "config": {}, "torch": True,
     "broken": "import torch; print(torch.nn.functional.cross_entropy(torch.zeros(1,2),torch.tensor([3])))",
     "fixed": "import torch; loss=torch.nn.functional.cross_entropy(torch.zeros(1,2),torch.tensor([1])); assert torch.isfinite(loss); print(loss.item())"},
    {"id": "detached_loss", "expected": "no_grad", "config": {}, "torch": True,
     "broken": "import torch; x=torch.tensor(2.); (x*x).backward()",
     "fixed": "import torch; x=torch.tensor(2.,requires_grad=True); (x*x).backward(); assert x.grad.item()==4.; print(x.grad.item())"},
    {"id": "unequal_lengths", "expected": "shape_mismatch", "config": {}, "torch": True,
     "broken": "import torch; torch.stack([torch.ones(2),torch.ones(3)])",
     "fixed": "import torch; from torch.nn.utils.rnn import pad_sequence; x=pad_sequence([torch.ones(2),torch.ones(3)],batch_first=True); assert x.shape==(2,3); print(tuple(x.shape))"},
    {"id": "unknown_error", "expected": None, "config": {},
     "broken": "raise ValueError('unrecognized experiment condition')", "fixed": None},
]


def execute(python, source, cwd):
    result = subprocess.run([str(python), "-I", "-c", source], cwd=cwd,
                            capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()
    python = args.python.resolve()
    if not python.is_file():
        parser.error("Fixture Python interpreter does not exist")
    results = []
    for case in FIXTURES:
        with tempfile.TemporaryDirectory(prefix="diagnostic-fixture-") as folder:
            failed = execute(python, case["broken"], folder)
            payload = {"log": failed["stdout"] + failed["stderr"],
                       "experiment_config": case["config"],
                       "environment": {"versions": {"torch": "2.5.1+cu121"}}}
            diagnosis = asyncio.run(inspect_experiment(payload, None))
            codes = [f["code"] for f in diagnosis["findings"]]
            repaired = execute(python, case["fixed"], folder) if case["fixed"] else None
            passed = failed["exit_code"] != 0 and (
                case["expected"] in codes if case["expected"] else not codes)
            passed = passed and (repaired is None or repaired["exit_code"] == 0)
            results.append({"id": case["id"], "passed": passed,
                            "input": payload, "expected": case["expected"],
                            "failed_run": failed, "diagnosis": diagnosis, "repaired_run": repaired,
                            "origin": "authored_cpu_fixture", "gpu_used": False})
            print(case["id"], "PASS" if passed else "FAIL", flush=True)
    report = {"passed": sum(r["passed"] for r in results), "total": len(results),
              "scope": "开发回归用例，不是独立终审集；修复为预先编写，不是模型自动生成。",
              "model_called": False, "results": results}
    (ROOT / "reports/fixtures.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
