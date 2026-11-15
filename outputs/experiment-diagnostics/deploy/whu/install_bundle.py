"""Python 3.9-compatible, offline CPU-job installer for the prepared WHU bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_members(members, limit=1500000000):
    if len(members) > 50000 or sum(member.size for member in members) > limit:
        raise ValueError("Archive exceeds prepared-bundle resource limits")
    names, links = set(), set()
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name:
            raise ValueError("Unsafe archive path")
        normalized = str(name)
        if normalized in names:
            raise ValueError("Duplicate archive path")
        names.add(normalized)
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError("Unsupported archive member type")
        if member.issym() or member.islnk():
            links.add(normalized)
            value = member.linkname
            if value.startswith("/") or "\\" in value or ":" in value:
                raise ValueError("Absolute or foreign archive link")
            target = posixpath.normpath(posixpath.join(str(name.parent), value) if member.issym() else value)
            if target == ".." or target.startswith("../"):
                raise ValueError("Archive link escapes destination")
    for name in names:
        if any(str(parent) in links for parent in PurePosixPath(name).parents):
            raise ValueError("Archive member traverses another archived link")


def extract_payload(archive, destination, expected):
    if destination.exists():
        marker = destination / ".payload-sha256"
        if not marker.is_file() or marker.read_text().strip() != expected:
            raise ValueError("Existing unrecognized installation directory: " + str(destination))
        return
    temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=str(destination.parent)))
    with tarfile.open(str(archive)) as package:
        members = package.getmembers()
        validate_members(members)
        # Python 3.9 lacks the data filter; paths/types/link parents are checked above.
        package.extractall(str(temporary), members=members)
    for path in temporary.rglob("*"):
        if path.is_symlink():
            try:
                path.resolve().relative_to(temporary.resolve())
            except ValueError:
                raise ValueError("Extracted link escapes destination")
    (temporary / ".payload-sha256").write_text(expected + "\n")
    temporary.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "bundle-manifest.json").read_text(encoding="utf-8"))
    for item in manifest["payloads"]:
        if item["file"] not in {"python312.tar.gz", "source.tar.gz", "wheels.tar.gz"}:
            raise ValueError("Unexpected payload name")
        path = ROOT / item["file"]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("Payload digest mismatch: " + item["file"])
    if not args.install:
        print(json.dumps({"payloads_verified": True, "install_executed": False,
            "python_target": "3.12.15", "gpu_requested": False}, indent=2))
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Prepared Python targets Linux x86_64 only")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Submit whu-setup.sbatch; do not install/run tests directly on the login node")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise ValueError("This prepared wheel profile requires glibc >= 2.28")
    state = ROOT / "state"
    state.mkdir(exist_ok=True)
    import fcntl
    with (state / ".setup.lock").open("a+") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        directories = {}
        for item in manifest["payloads"]:
            name = item["file"].split(".")[0]
            destination = state / (name + "-" + item["sha256"][:12])
            extract_payload(ROOT / item["file"], destination, item["sha256"])
            directories[name] = destination
        base_python = directories["python312"] / "python/bin/python3.12"
        application = directories["source"]
        launcher = application / "outputs/experiment-diagnostics/launch.py"
        bootstrap = application / "outputs/experiment-diagnostics/bootstrap_runtime.py"
        wheelhouse = directories["wheels"] / "linux-wheelhouse"
        environment = {key: value for key, value in os.environ.items()
            if not key.startswith(("DIAG_", "DEER_FLOW_")) and key not in {"PYTHONPATH", "PYTHONHOME"}}
        environment.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PIP_NO_INDEX="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CACHE_DIR=str(state / "pip-cache"),
            XDG_CACHE_HOME=str(state / "cache"))
        subprocess.run([str(base_python), "-c", "import ssl,sqlite3,venv; print('Independent Python available')"],
            check=True, env=environment)
        subprocess.run([str(base_python), str(bootstrap), "--install", "--wheelhouse", str(wheelhouse)],
            check=True, env=environment, cwd=str(application))
        python = application / "work/runtime-py312/bin/python"
        receipt = {"status": "framework_installed", "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "framework_python": str(python), "launcher": str(launcher), "application": str(application),
            "gpu_used": False, "api_called": False, "torch_environment_installed": False}
        (state / "installation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
