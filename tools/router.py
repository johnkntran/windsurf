"""
API router for the tools demonstration.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any

from .agent import ToolCallingAgent
from .models import ToolCall, StructuredOutput


# Create router
router = APIRouter(prefix="/tools", tags=["tools"])

# Initialize the agent
agent = ToolCallingAgent()


# Request/Response models
class ToolCallRequest(BaseModel):
    name: str = Field(..., description="Name of the tool to call")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class ToolCallResponse(BaseModel):
    result: Any = Field(..., description="Result of the tool call")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    chat_history: list[ChatMessage] = Field(default_factory=list, description="Chat history")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls made by the agent")
    final_answer: str | None = Field(None, description="Final answer if no tools were used")


# Endpoints
@router.post("/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest) -> ToolCallResponse:
    """Call a specific tool directly."""
    try:
        result = await agent.tool_manager.execute_tool(
            tool_name=request.name,
            tool_input=request.args
        )
        return ToolCallResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with the agent, which can use tools as needed."""
    try:
        # Convert chat history to the format expected by the agent
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.chat_history
        ]

        # Get the agent's response
        result = await agent.run(
            query=request.message,
            chat_history=chat_history
        )

        # Parse the output
        structured_output = agent.parse_structured_output(result["output"])

        return ChatResponse(
            response=result["output"],
            tool_calls=structured_output.tool_calls,
            final_answer=structured_output.final_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def list_available_tools() -> dict[str, Any]:
    """List all available tools and their descriptions."""
    tools = agent.tool_manager.get_tools()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": {
                    "type": "object",
                    "properties": {
                        name: {"type": "string"}  # Simplified for now
                        for name in tool.args_schema.schema()["properties"].keys()
                    },
                    "required": tool.args_schema.schema().get("required", [])
                } if hasattr(tool, 'args_schema') else {}
            }
            for tool in tools
        ]
    }
