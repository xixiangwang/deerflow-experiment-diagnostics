"""Tool selection and evidence-based termination for the bounded recipe route."""
import json

from langchain.agents.middleware import hook_config
from langchain_core.messages import AIMessage

from . import recipe_tools as tools
from .material_policy import ReadOnlyMaterialPolicy
from .runtime import create_api_client

TOOL_NAMES = ("inspect_recipe", "change_recipe", "search_recipe_reference")
MAX_MODEL_CALLS = 10
REPORT_SOURCE = "worker_label_policy"


def label_block_report():
    with tools.LOCK:
        if not tools.blocked_by_labels():
            return None
        last = tools.STATE["last"]
        problem = last["label_problem"]
        code = problem["code"]
        if code == "label_shape_mismatch":
            detail = f'输入张量形状为 {problem["input_shape"]}，标签张量形状为 {problem["label_shape"]}。'
        elif code == "label_out_of_range":
            detail = f'参与损失的标签 {problem["target"]} 超出词表范围 0 到 {problem["vocab_size"] - 1}。'
        else:
            detail = "因果语言建模移位后没有参与损失的监督标签。"
        return {"status": "unresolved", "blocked_by_label_policy": True,
                "observations": [detail, "操作者要求保留原标签。请核对标注后提交更正的配置。"],
                "changes": last["changes"], "final_run_id": last["run_id"],
                "final_revision": last["revision"], "final_recipe_sha256": last["recipe_sha256"],
                "label_policy_changed": False, "label_problem": problem,
                "limitations": ["固定随机微型 Qwen2 的 CPU 验证在前向阶段停止。"]}


class RecipePolicy(ReadOnlyMaterialPolicy):
    allowed = set(TOOL_NAMES)

    def __init__(self, max_calls=MAX_MODEL_CALLS):
        super().__init__()
        self.max_calls = max_calls

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        report = label_block_report()
        if report is not None:
            message = AIMessage(content=json.dumps(report, ensure_ascii=False),
                                additional_kwargs={"recipe_report_source": REPORT_SOURCE})
            return {"jump_to": "end", "messages": [message]}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    def available(self, request):
        if request.state.get("run_model_call_count", 0) >= self.max_calls - 1:
            return []
        with tools.LOCK:
            last = tools.STATE.get("last")
            if last and last["status"] == "passed":
                return []
            names = {"inspect_recipe"} if last is None else {"inspect_recipe", "change_recipe"}
            if last and tools.reference_available():
                names.add("search_recipe_reference")
        return [tool for tool in request.tools if tool.name in names]

    def wrap_model_call(self, request, handler):
        return handler(request.override(tools=self.available(request)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(tools=self.available(request)))


def create_recipe_client(run_id):
    return create_api_client(run_id, "deerflow_case.recipe_tools", TOOL_NAMES,
                             "declarative-experiment", MAX_MODEL_CALLS, scenario_policy=RecipePolicy())


def is_worker_report(message):
    return (message.get("additional_kwargs") or {}).get("recipe_report_source") == REPORT_SOURCE
