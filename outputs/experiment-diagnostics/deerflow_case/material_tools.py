"""Per-process read-only material session. No arbitrary path or execution tool is exposed."""
import copy
from typing import Literal
from langchain_core.tools import tool
from experiment_diagnostics.materials import inspect_materials
from experiment_diagnostics.knowledge import retrieve

_BUNDLE = None
TRACE = []


def bind_materials(bundle):
    global _BUNDLE
    _BUNDLE = copy.deepcopy(bundle)
    TRACE.clear()


@tool
def inspect_submitted_materials() -> dict:
    """Read the caller-selected, redacted logs/config/environment/code; run static checks only.

    Returns numbered evidence. Does not import or execute submitted code, apply a repair,
    or validate a user's experiment. Unknown errors remain hypotheses.
    """
    if _BUNDLE is None:
        raise ValueError("No material session")
    result = inspect_materials(_BUNDLE)
    TRACE.append({"tool": "inspect_submitted_materials", "result": result})
    return result


@tool
def search_material_references(query: str, package: Literal["peft", "pytorch", "transformers"],
                               allow_version_fallback: bool = False, diversify_sources: bool = False) -> dict:
    """Search official evidence matching submitted environment versions.

    If no compatible document exists, return no evidence. Explicit fallback may retrieve
    other versions, which remain unverified references, not proof of compatibility.
    Optional source diversity reduces repeated documents but can lose useful detail;
    it is off by default and should not be assumed to improve answer quality.
    """
    if _BUNDLE is None:
        raise ValueError("No material session")
    versions = _BUNDLE["materials"]["environment"].get("versions", {})
    version = versions.get(package, versions.get("torch") if package == "pytorch" else None)
    result = retrieve(query, package=package, top_k=3,
                      versions={package: version} if version and not allow_version_fallback else {},
                      diversify_sources=diversify_sources)
    result["requested_version"] = version
    result["version_fallback_allowed"] = allow_version_fallback
    TRACE.append({"tool": "search_material_references", "args": {"query": query, "package": package,
                  "allow_version_fallback": allow_version_fallback, "diversify_sources": diversify_sources}, "result": result})
    return result
