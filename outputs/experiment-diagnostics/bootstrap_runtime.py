"""Restore pinned framework dependencies in a project-owned Python 3.12 environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
MARKER = ".deerflow-runtime.json"
OWNER = "deerflow-experiment-diagnostics-runtime-v1"


def validate_target(target):
    target = Path(target).absolute()
    if target.is_symlink():
        raise ValueError("Runtime directory cannot be a symbolic link")
    target = target.resolve()
    if ROOT.is_relative_to(target) or target.is_relative_to(ROOT) or target in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        raise ValueError("Choose a separate runtime directory outside the source and active Python environment")
    if target.exists() and any(target.iterdir()):
        marker = target / MARKER
        if not marker.is_file() or marker.is_symlink():
            raise ValueError("Refusing to modify a nonempty environment not created by this bootstrap")
        if json.loads(marker.read_text(encoding="utf-8")).get("owner") != OWNER:
            raise ValueError("Runtime ownership marker does not match")
    return target


def choose_lock():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The verified locks target Python 3.12; use a Python 3.12 interpreter")
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise ValueError("Current locks target x86_64 only")
    if sys.platform == "win32":
        name = "requirements-windows-py312.lock"
    elif sys.platform == "linux" and platform.libc_ver()[0] == "glibc":
        name = "requirements-linux-py312.lock"
    else:
        raise ValueError("Current locks support Windows x86_64 and glibc Linux x86_64 only")
    path = ROOT / "deploy/agent" / name
    if not path.is_file():
        raise ValueError("Missing platform lock: " + name)
    return path


def clean_environment():
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(("DIAG_", "DEER_FLOW_")) and key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8")
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=PROJECT / "work/runtime-py312")
    parser.add_argument("--install", action="store_true", help="Create/restore a project-owned environment; otherwise show plan only")
    parser.add_argument("--uv", type=Path, help="Optional existing uv executable; default installer is pip")
    parser.add_argument("--wheelhouse", type=Path, help="Use a local wheel directory without contacting an index")
    parser.add_argument("--torch-python", type=Path, help="Optional existing training Python for the fixed CPU integration check")
    args = parser.parse_args()
    try:
        target, lock = validate_target(args.venv), choose_lock()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = ([str(args.uv.resolve()), "pip", "sync", str(lock), "--python", str(python), "--require-hashes",
                "--only-binary", ":all:", "--cache-dir", str(PROJECT / "work/host-cache")] if args.uv else
               [str(python), "-m", "pip", "install", "--require-hashes", "--only-binary=:all:", "-r", str(lock)])
    if args.wheelhouse:
        wheelhouse = args.wheelhouse.resolve()
        if not wheelhouse.is_dir():
            parser.error("--wheelhouse must be an existing directory")
        install.extend(["--no-index", "--find-links", str(wheelhouse)])
        if args.uv:
            install.append("--offline")
    plan = {"runtime": str(target), "python": str(python), "lock": lock.name,
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "install_command": install,
            "offline_wheelhouse": bool(args.wheelhouse),
            "modifies_existing_training_environment": False, "calls_model_api": False}
    if not args.install:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    created = not (target / MARKER).is_file()
    target.mkdir(parents=True, exist_ok=True)
    (target / MARKER).write_text(json.dumps({"owner": OWNER, "python": "3.12"}), encoding="utf-8")
    if not python.is_file():
        print("Creating isolated runtime", flush=True)
        venv.EnvBuilder(with_pip=args.uv is None).create(target)
    environment = clean_environment()
    environment["UV_PYTHON_INSTALL_DIR"] = str(PROJECT / "work/uv-managed")
    logs = PROJECT / "work/bootstrap-logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "install.log").open("w", encoding="utf-8") as output:
        print("Installing hash-locked dependencies", flush=True)
        subprocess.run(install, check=True, env=environment, stdout=output, stderr=subprocess.STDOUT)
    checks = []
    commands = [["doctor"], ["tests"]]
    if args.torch_python:
        commands.append(["self-test", "--torch-python", str(args.torch_python.resolve())])
    for arguments in commands:
        print("Checking", arguments[0], flush=True)
        process = subprocess.run([str(python), str(ROOT / "launch.py"), *arguments], env=environment,
            cwd=PROJECT, capture_output=True, encoding="utf-8", errors="replace", timeout=180)
        (logs / (arguments[0] + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        checks.append({"command": arguments[0], "exit_code": process.returncode})
        if process.returncode:
            raise RuntimeError("Runtime check failed; see " + str(logs / (arguments[0] + ".log")))
    report = {**plan, "created_fresh_environment": created, "platform": platform.platform(), "checks": checks,
              "api_called": False, "linux_executed": sys.platform == "linux",
              "scope": "Fresh dependency restore on this platform; not school-server or container deployment."}
    (ROOT / "reports/runtime-bootstrap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "fresh_environment": created, "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
