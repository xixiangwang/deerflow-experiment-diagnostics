"""A real local-model provider plugged into DeerFlow's normal model factory."""
import json
import os
from pathlib import Path
import subprocess
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

GENERATIONS = []


class LocalChatModel(BaseChatModel):
    model: str = "local-qwen-0.5b"
    _tools: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "local-transformers-qwen"

    def bind_tools(self, tools, **kwargs):
        clone = self.model_copy()
        clone._tools = [convert_to_openai_tool(tool) for tool in tools]
        return clone

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        wire = []
        for message in messages:
            role = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}[message.type]
            content = message.content
            if isinstance(content, list):
                if any(not isinstance(block, dict) or block.get("type") != "text" for block in content):
                    raise ValueError("Local text-only provider received non-text content")
                content = "\n".join(block["text"] for block in content)
            item = {"role": role, "content": content}
            if message.type == "ai" and message.tool_calls:
                item["tool_calls"] = [{"type": "function", "id": c["id"], "function": {
                    "name": c["name"], "arguments": json.dumps(c["args"])}} for c in message.tool_calls]
            if message.type == "tool":
                item.update(tool_call_id=message.tool_call_id, name=message.name)
            wire.append(item)
        request = {"messages": wire, "tools": self._tools,
                   "model_path": os.environ["DIAG_LOCAL_MODEL"]}
        result = subprocess.run([os.environ["DIAG_TORCH_PYTHON"], "-I",
                                 str(Path(__file__).with_name("local_model_worker.py"))],
                                input=json.dumps(request, ensure_ascii=False),
                                capture_output=True, encoding="utf-8", errors="replace", timeout=120)
        if result.returncode:
            raise RuntimeError("Local provider failed: " + result.stderr[-1500:])
        generated = json.loads(result.stdout)
        allowed = {t["function"]["name"] for t in self._tools}
        if any(c["name"] not in allowed for c in generated["tool_calls"]):
            raise ValueError("Model requested an unknown tool")
        GENERATIONS.append(generated)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=generated["content"], tool_calls=generated["tool_calls"]))])
