"""
Agent implementation for tool-calling demonstration.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

from .models import StructuredOutput, ToolCall
from .tools import ToolManager


class ToolCallingAgent:
    """An agent that can use tools based on user queries."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        """Initialize the agent with a language model and tools."""
        self.tool_manager = ToolManager()
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.verbose = verbose
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        """Create and configure the agent with tools."""
        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that can use tools to answer questions.
            When you need to use a tool, provide the tool name and input.
            If you don't need to use any tools, provide a final answer directly.

            Available tools:
            {tools}

            Use the following format for tool calls:
            ```json
            {{
                "thought": "your thought process",
                "tool_calls": [
                    {{
                        "name": "tool_name",
                        "args": {{"arg1": "value1", "arg2": "value2"}}
                    }}
                ],
                "final_answer": "final answer if no tools are needed"
            }}
            ```

            If you need to use multiple tools, include multiple tool calls in the array.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tool_manager.get_tools(),
            prompt=prompt,
        )

        # Create the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tool_manager.get_tools(),
            verbose=self.verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    async def run(self, query: str, chat_history: list | None = None) -> dict[str, Any]:
        """Run the agent with the given query and chat history."""
        if chat_history is None:
            chat_history = []

        # Prepare the input for the agent
        input_data = {
            "input": query,
            "chat_history": chat_history,
        }

        # Run the agent
        try:
            result = await self.agent.ainvoke(input_data)

            # Process the result
            output = {
                "output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
            }

            return output
        except Exception as e:
            return {
                "output": f"An error occurred: {str(e)}",
                "intermediate_steps": [],
            }

    def parse_structured_output(self, text: str) -> StructuredOutput:
        """Parse the structured output from the model's response."""
        try:
            # Try to parse as JSON first
            import json
            data = json.loads(text)

            # Convert to the structured output model
            return StructuredOutput(
                thought=data.get("thought", ""),
                tool_calls=[
                    ToolCall(
                        name=tool_call["name"],
                        args=tool_call.get("args", {})
                    )
                    for tool_call in data.get("tool_calls", [])
                ],
                final_answer=data.get("final_answer"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, return the text as the final answer
            return StructuredOutput(
                thought="Could not parse structured output",
                tool_calls=[],
                final_answer=text,
            )
