"""Operator policy for the submitted-material route, using standard middleware hooks."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


class ReadOnlyMaterialPolicy(AgentMiddleware):
    allowed = {"inspect_submitted_materials", "search_material_references"}

    def wrap_model_call(self, request, handler):
        tools = [t for t in request.tools if getattr(t, "name", None) in self.allowed]
        return handler(request.override(tools=tools))

    async def awrap_model_call(self, request, handler):
        tools = [t for t in request.tools if getattr(t, "name", None) in self.allowed]
        return await handler(request.override(tools=tools))

    def wrap_tool_call(self, request, handler):
        if request.tool_call.get("name") not in self.allowed:
            return ToolMessage(content="Tool not permitted in the read-only material route.",
                               tool_call_id=request.tool_call["id"])
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call.get("name") not in self.allowed:
            return ToolMessage(content="Tool not permitted in the read-only material route.",
                               tool_call_id=request.tool_call["id"])
        return await handler(request)
