"""Usage: python -m experiment_diagnostics input.json [--output report.json]."""
import argparse
import asyncio
import json
from pathlib import Path
from . import inspect_experiment


def main():
    parser = argparse.ArgumentParser(description="Read-only experiment log diagnosis")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input.stat().st_size > 512000:
        parser.error("Input file must be at most 512000 bytes")
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        parser.error("Input JSON must be an object")
    result = asyncio.run(inspect_experiment(payload, None))
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
