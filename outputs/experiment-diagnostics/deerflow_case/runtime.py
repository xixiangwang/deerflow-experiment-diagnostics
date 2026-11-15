"""Shared scenario runtime setup using official client/config/middleware extension points."""
import os
from pathlib import Path
from urllib.parse import urlparse
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parent.parent / "work"


def create_api_client(run_id, tool_module, tool_names, skill, model_call_limit=8, *, scenario_policy=None):
    missing = [key for key in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL") if not os.environ.get(key)]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    WORK.mkdir(exist_ok=True)
    config = yaml.safe_load((ROOT / "deerflow-test.yaml").read_text(encoding="utf-8"))
    model = {"name": "scenario-model", "use": "langchain_openai:ChatOpenAI", "model": "$DIAG_MODEL",
             "api_key": "$DIAG_API_KEY", "base_url": "$DIAG_BASE_URL", "temperature": 0,
             "max_retries": 1, "timeout": 60, "max_tokens": 2200}
    if urlparse(os.environ["DIAG_BASE_URL"]).hostname == "api.deepseek.com":
        model["extra_body"] = {"thinking": {"type": "disabled"}}
    config["models"] = [model]
    config["tools"] = [{"name": name, "group": "diagnosis", "use": tool_module + ":" + name} for name in tool_names]
    path = WORK / (run_id + ".yaml")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    os.environ.update(DEER_FLOW_PROJECT_ROOT=str(ROOT), DEER_FLOW_CONFIG_PATH=str(path),
                     DEER_FLOW_HOME=str(WORK / run_id), DEER_FLOW_SKILLS_PATH=str(ROOT / "deerflow_case/skills"))
    os.environ.setdefault("DIAG_KNOWLEDGE_INDEX", str(ROOT / "knowledge/index.sqlite3"))
    from deerflow.client import DeerFlowClient
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from .material_policy import ReadOnlyMaterialPolicy

    if scenario_policy is None:
        class ScenarioToolPolicy(ReadOnlyMaterialPolicy):
            allowed = set(tool_names)

        scenario_policy = ScenarioToolPolicy()

    client = DeerFlowClient(config_path=str(path), model_name="scenario-model", thinking_enabled=False,
                           subagent_enabled=False, available_skills={skill},
                           middlewares=[scenario_policy, ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end")])
    assert client._app_config.get_model_config("scenario-model").use == "langchain_openai:ChatOpenAI"
    return client


def final_model_text(events):
    chunks = {}
    for event in events:
        data = event["data"]
        if event["type"] == "messages-tuple" and data.get("type") == "ai" and data.get("content"):
            chunks.setdefault(data.get("id", "answer"), []).append(data["content"])
    return "".join(next(reversed(chunks.values()))) if chunks else ""
