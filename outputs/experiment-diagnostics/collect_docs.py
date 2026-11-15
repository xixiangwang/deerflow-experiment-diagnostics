"""Fetch only allowlisted official, version-tagged source documents."""
import concurrent.futures
import base64
import hashlib
import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
SPECS = [
    ("pytorch", "pytorch/pytorch", "v2.5.1", [
        "docs/source/notes/cuda.rst", "docs/source/notes/amp_examples.rst",
        "docs/source/notes/faq.rst", "docs/source/notes/numerical_accuracy.rst",
        "docs/source/notes/randomness.rst", "docs/source/notes/serialization.rst",
        "docs/source/notes/autograd.rst", "docs/source/notes/windows.rst",
    ]),
    ("transformers", "huggingface/transformers", "v4.46.3", [
        "docs/source/en/quantization/bitsandbytes.md", "docs/source/en/quantization/overview.md",
        "docs/source/en/chat_templating.md", "docs/source/en/pad_truncation.md",
        "docs/source/en/main_classes/data_collator.md", "docs/source/en/main_classes/trainer.md",
        "docs/source/en/perf_train_gpu_one.md", "docs/source/en/troubleshooting.md",
    ]),
    ("peft", "huggingface/peft", "v0.13.2", [
        "docs/source/developer_guides/quantization.md", "docs/source/developer_guides/lora.md",
        "docs/source/developer_guides/troubleshooting.md", "docs/source/quicktour.md",
        "docs/source/package_reference/lora.md", "docs/source/developer_guides/checkpoint.md",
    ]),
    ("peft", "huggingface/peft", "v0.21.2", [
        "docs/source/developer_guides/quantization.md", "docs/source/developer_guides/custom_models.md",
        "docs/source/developer_guides/troubleshooting.md", "docs/source/quicktour.md",
        "docs/source/package_reference/lora.md", "docs/source/developer_guides/checkpoint.md",
        "docs/source/developer_guides/memory_efficient_training.md",
        "src/peft/tuners/tuners_utils.py",
    ]),
    ("transformers", "huggingface/transformers", "v5.18.0", [
        "docs/source/en/quantization/bitsandbytes.md", "docs/source/en/quantization/overview.md",
        "docs/source/en/chat_templating.md", "docs/source/en/main_classes/tokenizer.md",
        "docs/source/en/main_classes/data_collator.md", "docs/source/en/main_classes/trainer.md",
        "docs/source/en/training.md", "docs/source/en/troubleshooting.md",
        "docs/source/en/model_memory_anatomy.md", "docs/source/en/mixed_precision_training.md",
        "docs/source/en/optimization_overview.md",
        "src/transformers/models/qwen2/modeling_qwen2.py",
    ]),
    ("transformers", "huggingface/transformers", "v4.26.1", [
        "src/transformers/trainer.py",
    ]),
    ("transformers", "huggingface/transformers", "v4.28.0", [
        "src/transformers/generation/utils.py",
    ]),
]


def resolve_commit(repository, tag):
    response = requests.get(f"https://api.github.com/repos/{repository}/git/ref/tags/{tag}", timeout=(8, 20))
    response.raise_for_status()
    obj = response.json()["object"]
    for _ in range(3):
        if not re.fullmatch(r"[0-9a-f]{40}", obj["sha"]):
            raise ValueError("Invalid upstream object SHA")
        if obj["type"] == "commit":
            return obj["sha"]
        if obj["type"] != "tag":
            break
        response = requests.get(f"https://api.github.com/repos/{repository}/git/tags/{obj['sha']}", timeout=(8, 20))
        response.raise_for_status()
        obj = response.json()["object"]
    raise ValueError("Could not resolve tag to a commit")


def fetch(item):
    package, repository, version, path, commit = item
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
    destination = ROOT / "knowledge" / "raw" / package / version / path.replace("/", "__")
    api_url = f"https://api.github.com/repos/{repository}/contents/{path}?ref={commit}"
    for attempt in range(2):
        try:
            response = requests.get(api_url, timeout=(8, 20))
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("encoding") != "base64":
                raise ValueError("Expected GitHub Contents API base64 encoding")
            content = base64.b64decode(metadata["content"])
            if len(content) < 100 or len(content) > 2000000:
                raise ValueError("Unexpected document length")
            # Decode now to reject HTML/binary responses, but retain original bytes.
            decoded = content.decode("utf-8")
            if decoded.lstrip().lower().startswith("<!doctype html"):
                raise ValueError("HTML response instead of source document")
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if blob != metadata["sha"]:
                raise ValueError("Git blob digest mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            return {"id": package + ":" + version + ":" + path,
                    "package": package, "version": version.removeprefix("v"),
                    "url": f"https://github.com/{repository}/blob/{commit}/{path}", "source_commit": commit,
                    "download_url": url, "fetch_url": api_url, "git_blob_sha": metadata["sha"],
                    "file": destination.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content), "chars": len(decoded),
                    "fetched_at": datetime.now(timezone.utc).isoformat(), "status": "ok"}
        except (requests.RequestException, UnicodeError, ValueError) as exc:
            if attempt == 1:
                return {"package": package, "path": path, "status": "failed", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refetch verified cached files as well")
    args = parser.parse_args()
    manifest_path = ROOT / "knowledge/manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    cached = {}
    for entry in existing:
        if entry.get("status") != "ok":
            continue
        source = (ROOT / entry["file"]).resolve()
        if source.is_relative_to(ROOT.resolve()) and source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]:
            cached[entry["id"]] = entry
    items, retained = [], []
    for package, repository, version, paths in SPECS:
        missing = []
        for path in paths:
            identifier = package + ":" + version + ":" + path
            if identifier in cached and not args.refresh:
                retained.append(cached[identifier])
            else:
                missing.append(path)
        if missing:
            commit = resolve_commit(repository, version)
            items.extend((package, repository, version, path, commit) for path in missing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(retained)
        for entry in pool.map(fetch, items):
            entries.append(entry)
            print(entry["status"], entry.get("id", entry.get("path")), flush=True)
    folder = ROOT / "knowledge"
    folder.mkdir(exist_ok=True)
    good = [item for item in entries if item["status"] == "ok"]
    print(json.dumps({"downloaded": len(good), "failed": len(entries) - len(good),
                      "chars": sum(item["chars"] for item in good)}, ensure_ascii=False))
    for item in entries:
        if item["status"] != "ok":
            print("FAILED", item["path"], item["error"])
    if len(good) != len(entries):
        raise SystemExit(1)
    temporary = manifest_path.with_suffix(".building.json")
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
