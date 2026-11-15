"""Verify the extension through the actual upstream host loader, no LLM."""
import asyncio
import json
from pathlib import Path

scanner = (Path(__file__).parent.parent / "deer-flow/backend/packages/harness/deerflow/skills/skillscan/orchestrator.py")
if not scanner.exists():
    raise SystemExit("BLOCKED: upstream skillscan/orchestrator.py is missing. See STATUS.md; review local file/security handling before restoring and retrying.")
from deerflow.extensions.loader import ExtensionSpec, load_extensions

loaded, diagnostics = load_extensions([
    ExtensionSpec(use="experiment_diagnostics:install", required=True)
])
assert not diagnostics, diagnostics
assert len(loaded.plugins) == 1
source, plugin = loaded.plugins[0]
assert plugin.namespace == "research.experiment_diagnostics"
tool = plugin.tools[0]
result = asyncio.run(tool.handler({"log": "CUDA out of memory"}, None))
assert result["findings"][0]["code"] == "cuda_oom"
report = {"upstream_host_loader": "passed", "plugin": plugin.namespace,
          "tool": tool.name, "result": result, "model_called": False}
path = Path(__file__).parent / "reports" / "host.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("PASS: actual DeerFlow loader registered and invoked the diagnostic extension")
