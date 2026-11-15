"""Deterministic integration-test double. This is explicitly NOT a real LLM."""
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

CALLS = []
SKILL_VISIBLE = []


class IntegrationTestModel(BaseChatModel):
    model: str = "deterministic-integration-test"

    @property
    def _llm_type(self):
        return "deterministic-integration-test-NOT-LLM"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        SKILL_VISIBLE.append(any("DIAGNOSTIC_SKILL_V1" in str(message.content) for message in messages))
        CALLS.append([{"type": message.type, "name": message.name,
                       "content": str(message.content)[:10000]} for message in messages
                      if isinstance(message, ToolMessage)])
        observations = {message.name: message for message in messages if isinstance(message, ToolMessage)}
        if "inspect_lora_experiment" not in observations:
            name, arguments = "inspect_lora_experiment", {"case_id": "tiny_qwen2_lora"}
        elif "search_experiment_docs" not in observations:
            name, arguments = "search_experiment_docs", {"query": "target_modules named_modules inspect module names model architecture"}
        elif "validate_lora_fix" not in observations:
            inspection = json.loads(observations["inspect_lora_experiment"].content)
            choices = [name for name in ("q_proj", "v_proj") if name in inspection["available_linear_modules"]]
            name, arguments = "validate_lora_fix", {"case_id": "tiny_qwen2_lora", "target_modules": choices}
        else:
            inspection = json.loads(observations["inspect_lora_experiment"].content)
            docs = json.loads(observations["search_experiment_docs"].content)
            validation = json.loads(observations["validate_lora_fix"].content)
            answer = ("# DeerFlow 集成测试报告\n\n"
                      "本报告由确定性测试替身生成，不是真实大模型诊断。\n\n"
                      f"实际故障：{inspection['error']}\n\n"
                      f"实际模块列表：{inspection['available_linear_modules']}\n\n"
                      f"测试修复：target_modules={validation['requested_target_modules']}。\n\n"
                      f"验证：{validation['status']}；完成 {validation['validation_steps']} 个优化步骤，"
                      f"梯度有限={validation['finite_gradients']}，参数改变={validation['parameters_changed']}。\n\n"
                      "资料版本与当前环境未必一致，不能仅凭检索结果认定兼容；本次结论由实际 CPU 重跑支撑。\n\n"
                      + "\n".join(hit["citation"] for hit in docs["hits"]))
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])
        message = AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": name + "-call"}])
        return ChatResult(generations=[ChatGeneration(message=message)])
