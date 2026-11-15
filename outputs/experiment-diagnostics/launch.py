"""Source-first launcher: uses an existing Python environment, no implicit installs."""
import argparse
import getpass
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
HARNESS = ROOT.parent / "deer-flow-v2.0/backend/packages/harness"
CONTRACT = ROOT.parent / "deer-flow/backend/packages/extension-api"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="DeerFlow experiment diagnosis project")
    parser.add_argument("command", choices=["doctor", "materials", "multistep", "recipe", "recipe-check", "recipe-demo", "recipe-holdout", "recipe-holdout-preflight", "recipe-policy-check", "pretrained", "pretrained-check", "gpu-oom", "gpu-oom-agent", "evaluate", "compare", "compare-loops", "compare-public", "console", "collect", "index", "retrieval-check", "self-test", "tests"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--preserve-labels", action="store_true", help="Keep supplied labels unchanged in recipe diagnosis")
    parser.add_argument("--torch-python", type=Path)
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--slurm-profile", type=Path, help="Operator SSH/Slurm profile for the fixed pretrained trial")
    parser.add_argument("--knowledge-index", type=Path, help="Optional saved index for an earlier corpus")
    parser.add_argument("--diversify-sources", action="store_true", help="Optional source diversity for retrieval-check")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    if args.preserve_labels and args.command != "recipe":
        parser.error("--preserve-labels is for recipe")
    if args.slurm_profile:
        if args.command not in {"pretrained", "gpu-oom"}:
            parser.error("--slurm-profile supports pretrained and gpu-oom")
        from deerflow_case.slurm_pretrained import SlurmPretrainedExecutor
        SlurmPretrainedExecutor(args.slurm_profile)
        os.environ["DIAG_SLURM_PROFILE"] = str(args.slurm_profile.resolve())
    if sys.version_info < (3, 12):
        parser.error("Use Python 3.12 or newer")
    if args.input:
        args.input = args.input.resolve()
    if args.torch_python:
        args.torch_python = args.torch_python.resolve()
    if not (HARNESS / "deerflow/client.py").is_file() or not (CONTRACT / "deerflow_extension_api/__init__.py").is_file():
        parser.error("Install the pinned DeerFlow sources with python tools/setup_upstream.py")
    for path in (HARNESS, CONTRACT, ROOT):
        sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, (ROOT, HARNESS, CONTRACT)))
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if args.knowledge_index:
        selected_index = args.knowledge_index.resolve()
        if not selected_index.is_file():
            parser.error("--knowledge-index must be an existing SQLite index")
        os.environ["DIAG_KNOWLEDGE_INDEX"] = str(selected_index)
    (PROJECT / "work").mkdir(exist_ok=True)
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("yaml", "langchain", "langchain_openai", "langgraph", "jsonschema", "sqlalchemy")}
    if args.command == "doctor":
        print(json.dumps({"python": sys.version.split()[0], "dependencies": dependencies,
            "framework": str(HARNESS), "index_exists": (ROOT / "knowledge/index.sqlite3").is_file(),
            "api_key_present": bool(os.environ.get("DIAG_API_KEY")),
            "note": "This checks availability, not full official deployment or model quality"}, indent=2))
        if not all(dependencies.values()):
            raise SystemExit(1)
        return
    if not all(dependencies.values()):
        parser.error("Missing dependencies; install requirements-harness.txt into an isolated environment")
    if args.command in {"materials", "recipe", "recipe-check"} and (args.input is None or not args.input.exists()):
        parser.error(args.command + " requires --input")
    if args.command in {"recipe", "recipe-check"}:
        from deerflow_case.recipe_spec import read_recipe
        try:
            read_recipe(args.input)
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))
    gpu_commands = {"pretrained", "pretrained-check", "gpu-oom", "gpu-oom-agent"}
    if args.command in gpu_commands | {"multistep", "recipe", "recipe-demo", "recipe-holdout", "recipe-holdout-preflight", "recipe-policy-check", "evaluate", "compare-loops", "self-test"} and not args.slurm_profile:
        if args.torch_python:
            os.environ["DIAG_TORCH_PYTHON"] = str(args.torch_python)
        elif not os.environ.get("DIAG_TORCH_PYTHON"):
            if all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers")):
                os.environ["DIAG_TORCH_PYTHON"] = sys.executable
            else:
                parser.error("Specify --torch-python pointing to an existing torch/peft/transformers environment")
    if args.local_model:
        os.environ["DIAG_LOCAL_MODEL"] = str(args.local_model.resolve())
    if args.command in gpu_commands and not args.slurm_profile:
        model_path = os.environ.get("DIAG_LOCAL_MODEL")
        if not model_path or not (Path(model_path) / "config.json").is_file():
            parser.error("Specify --local-model with existing local Qwen2.5-0.5B model files; downloads are disabled")
        os.environ["DIAG_LOCAL_MODEL"] = str(Path(model_path).resolve())
    os.chdir(ROOT)
    if args.command == "tests":
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests")))
        raise SystemExit(not result.wasSuccessful())
    if args.command == "recipe-holdout-preflight":
        from deerflow_case.recipe_holdout import preflight
        raise SystemExit(not preflight())
    api_needed = args.command not in {"self-test", "tests", "collect", "index", "retrieval-check", "recipe-check", "recipe-demo", "pretrained-check", "gpu-oom"}
    had_key = bool(os.environ.get("DIAG_API_KEY"))
    if api_needed:
        os.environ["DIAG_BASE_URL"] = args.base_url
        os.environ["DIAG_MODEL"] = args.model
        if args.command != "materials" and not had_key:
            os.environ["DIAG_API_KEY"] = getpass.getpass("API key (not saved): ")
    scripts = {"materials": "analyze_materials.py", "multistep": "run_multistep.py",
               "recipe": "run_recipe.py", "recipe-check": "check_recipe.py", "recipe-demo": "demo_recipe.py",
               "recipe-holdout": "compare_recipe_holdout.py",
               "recipe-policy-check": "evaluate_recipe_policy.py",
               "collect": "collect_docs.py", "retrieval-check": "evaluate_retrieval.py",
               "index": "experiment_diagnostics/knowledge.py",
               "pretrained": "run_pretrained.py", "pretrained-check": "check_pretrained.py", "gpu-oom": "run_gpu_oom.py",
               "gpu-oom-agent": "run_gpu_oom_agent.py",
               "compare-loops": "compare_multistep.py", "console": "agent_server.py",
               "evaluate": "evaluate_deerflow.py", "compare": "compare_materials.py", "compare-public": "evaluate_public_issues.py",
               "self-test": "run_deerflow_integration.py"}
    script = ROOT / scripts[args.command]
    sys.argv = [str(script)] + ([str(args.input)] if args.command in {"materials", "recipe", "recipe-check"} else [])
    if args.preserve_labels:
        sys.argv.append("--preserve-labels")
    if args.command == "materials":
        sys.argv.append("--prompt-for-key")
    if args.command == "console":
        sys.argv.extend(["--host", args.host, "--port", str(args.port)])
    if args.command == "retrieval-check" and args.diversify_sources:
        sys.argv.append("--diversify-sources")
    try:
        if args.command == "index":
            from experiment_diagnostics.knowledge import build_index
            print(json.dumps(build_index(), ensure_ascii=False, indent=2))
        else:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        if api_needed and not had_key:
            os.environ.pop("DIAG_API_KEY", None)


if __name__ == "__main__":
    main()
