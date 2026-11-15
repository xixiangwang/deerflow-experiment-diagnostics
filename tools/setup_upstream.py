"""Fetch the two pinned DeerFlow packages used by the experiment runner."""

from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://codeload.github.com/bytedance/deer-flow/tar.gz"
PACKAGES = (
    ("deer-flow-v2.0", "7e7f0410797693cf882594555ba414e0361d4c6f", "backend/packages/harness"),
    ("deer-flow", "2a97ec3e589821e8435d57ef3f160ef6824b2a45", "backend/packages/extension-api"),
)


def main():
    outputs = ROOT / "outputs"
    for name, commit, subtree in PACKAGES:
        destination = outputs / name
        if (destination / ".upstream-commit").is_file():
            if (destination / ".upstream-commit").read_text().strip() == commit:
                if (destination / subtree / "pyproject.toml").is_file() and (destination / "LICENSE").is_file():
                    print(f"{name}: {commit[:12]} already installed")
                    continue
        if destination.exists():
            raise SystemExit(f"Existing directory requires review: {destination}")
        with tempfile.TemporaryDirectory(prefix="df-", dir=outputs) as directory:
            temporary = Path(directory)
            archive_path = temporary / "source.tar.gz"
            extracted = temporary / "source"
            with urlopen(f"{UPSTREAM}/{commit}", timeout=60) as source, archive_path.open("wb") as archive:
                shutil.copyfileobj(source, archive)
            prefix = f"deer-flow-{commit}/"
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive:
                    if member.name != prefix + "LICENSE" and not member.name.startswith(prefix + subtree + "/"):
                        continue
                    if member.isdir():
                        continue
                    relative = PurePosixPath(member.name.removeprefix(prefix))
                    if (not member.isreg() or relative.is_absolute() or ".." in relative.parts
                            or any("\\" in part or ":" in part for part in relative.parts)):
                        raise ValueError("Unexpected file in DeerFlow archive")
                    target = extracted.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
            if not ((extracted / subtree / "pyproject.toml").is_file() and (extracted / "LICENSE").is_file()):
                raise ValueError("DeerFlow archive is missing the requested package or license")
            (extracted / ".upstream-commit").write_text(commit + "\n", encoding="ascii")
            extracted.rename(destination)
            print(f"{name}: installed {commit[:12]}")


if __name__ == "__main__":
    main()
