# Building Agents #

import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

llm = OpenAI(model="gpt-4o-mini")

workflow = FunctionAgent(
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    llm=llm,
    tools=[multiply, add],
)

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())

# Using Existing Tools #

from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply, add])

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=OpenAI(model="gpt-4o-mini"),
    tools=finance_tools,
    system_prompt="You are a helpful assistant.",
)

async def main():
    response = await workflow.run(
        user_msg="What's the current stock price of MSFT?"
    )
    print(response)


# Maintaing State #

from llama_index.core.workflow import Context, JsonPickleSerializer, JsonSerializer
from llama_index.core.agent.workflow import AgentWorkflow

ctx = Context(workflow)
response = await workflow.run(user_msg="Hi, my name is Laurie!", ctx=ctx)
print(response)

response2 = await workflow.run(user_msg="What's my name?", ctx=ctx)
print(response2)

ctx_dict = ctx.to_dict(serializer=JsonSerializer())

restored_ctx = Context.from_dict(
    workflow, ctx_dict, serializer=JsonSerializer()
)

response3 = await workflow.run(user_msg="What's my name?", ctx=restored_ctx)

async def set_name(ctx: Context, name: str) -> str:
    state = await ctx.store.get("state")
    state["name"] = name
    await ctx.store.set("state", state)
    return f"Name set to {name}"

workflow = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"},
)

ctx = Context(workflow)

# check if it knows a name before setting it
response = await workflow.run(user_msg="What's my name?", ctx=ctx)
print(str(response))

response2 = await workflow.run(user_msg="My name is Laurie", ctx=ctx)
print(str(response2))

state = await ctx.store.get("state")
print("Name as stored in state: ", state["name"])

# Streaming Outputs and Events #

from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import AgentStream, AgentInput, AgentOutput, ToolCall, ToolCallResult
import os

tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

workflow = FunctionAgent(
    tools=tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information.",
)

handler = workflow.run(user_msg="What's the weather like in San Francisco?")

async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)
    elif isinstance(event, AgentInput):
        print("Agent input: ", event.input)  # the current input messages
        print("Agent name:", event.current_agent_name)  # the current agent name
    elif isinstance(event, AgentOutput):
        print("Agent output: ", event.response)  # the current full response
        print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
        print("Raw LLM response: ", event.raw)  # the raw llm api response
    elif isinstance(event, ToolCallResult):
        print("Tool called: ", event.tool_name)  # the tool name
        print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
        print("Tool output: ", event.tool_output)  # the tool output
print(str(await handler))

# Human in the Loop #

from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent, \
    Context

async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""

    # emit a waiter event (InputRequiredEvent here)
    # and wait until we see a HumanResponseEvent
    question = "Are you sure you want to proceed? "
    response = await ctx.wait_for_event(
        event_type=HumanResponseEvent,
        waiter_id=question,
        waiter_event=InputRequiredEvent(
            prefix=question,
            user_name="Laurie",
        ),
        requirements={"user_name": "Laurie"},
    )

    # act on the input from the event
    if response.response.strip().lower() == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."

workflow = FunctionAgent(
    system_prompt="You are a helpful assistant that can perform dangerous tasks.",
    tools=[dangerous_task],
    llm=llm,
)

handler = workflow.run(user_msg="I want to proceed with the dangerous task.")

async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        # capture keyboard input
        response = input(event.prefix)
        # send our response back
        handler.ctx.send_event(
            HumanResponseEvent(
                response=response,
                user_name=event.user_name,
            )
        )

response = await handler
print(str(response))

# Multi-Agent Workflows: AgentWorkflow - Linear Swarm Pattern #

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Search the web and record notes.",
    system_prompt="You are a researcher… hand off to WriteAgent when ready.",
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a markdown report from the notes.",
    system_prompt="You are a writer… ask ReviewAgent for feedback when done.",
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Reviews a report and gives feedback.",
    system_prompt="You are a reviewer…",  # etc.
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

resp = await agent_workflow.run(
    user_msg="Write me a report on the history of the web …"
)
print(resp)

# Multi-Agent Workflows: Orchestrator Agent - Sub-Agents as Tools #

import re
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context

'''
assume research_agent / write_agent / review_agent defined as before
except we really only need the `search_web` tool at a minimum
'''

async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(
        user_msg=f"Write some notes about the following: {prompt}"
    )

    state = await ctx.store.get("state")
    state["research_notes"].append(str(result))
    await ctx.store.set("state", state)

    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    state = await ctx.store.get("state")
    notes = state.get("research_notes", None)
    if not notes:
        return "No research notes to write from."

    user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: <report>...</report>:\n\n"

    # Add the feedback to the user message if it exists
    feedback = state.get("review", None)
    if feedback:
        user_msg += f"<feedback>{feedback}</feedback>\n\n"

    # Add the research notes to the user message
    notes = "\n\n".join(notes)
    user_msg += f"<research_notes>{notes}</research_notes>\n\n"

    # Run the write agent
    result = await write_agent.run(user_msg=user_msg)
    report = re.search(r"<report>(.*)</report>", str(result), re.DOTALL).group(
        1
    )
    state["report_content"] = str(report)
    await ctx.store.set("state", state)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    state = await ctx.store.get("state")
    report = state.get("report_content", None)
    if not report:
        return "No report content to review."

    result = await review_agent.run(
        user_msg=f"Review the following report: {report}"
    )
    state["review"] = result
    await ctx.store.set("state", state)

    return result


orchestrator = FunctionAgent(
    system_prompt=(
        "You are an expert in the field of report writing. "
        "You are given a user request and a list of tools that can help with the request. "
        "You are to orchestrate the tools to research, write, and review a report on the given topic. "
        "Once the review is positive, you should notify the user that the report is ready to be accessed."
    ),
    llm=orchestrator_llm,
    tools=[
        call_research_agent,
        call_write_agent,
        call_review_agent,
    ],
    initial_state={
        "research_notes": [],
        "report_content": None,
        "review": None,
    },
)

response = await orchestrator.run(
    user_msg="Write me a report on the history of the web …"
)
print(response)

# Multi-Agent Workflows: Custom Planner - DIY Prompting & Parsing #

import re
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from typing import Any, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

'''
Assume we created helper functions to call the agents
'''

PLANNER_PROMPT = """You are a planner chatbot.

Given a user request and the current state, break the solution into ordered <step> blocks.  Each step must specify the agent to call and the message to send, e.g.
<plan>
  <step agent=\"ResearchAgent\">search for …</step>
  <step agent=\"WriteAgent\">draft a report …</step>
  ...
</plan>

<state>
{state}
</state>

<available_agents>
{available_agents}
</available_agents>

The general flow should be:
- Record research notes
- Write a report
- Review the report
- Write the report again if the review is not positive enough

If the user request does not require any steps, you can skip the <plan> block and respond directly.
"""


class InputEvent(StartEvent):
    user_msg: Optional[str] = Field(default=None)
    chat_history: list[ChatMessage]
    state: Optional[dict[str, Any]] = Field(default=None)


class OutputEvent(StopEvent):
    response: str
    chat_history: list[ChatMessage]
    state: dict[str, Any]


class StreamEvent(Event):
    delta: str


class PlanEvent(Event):
    step_info: str


# Modelling the plan
class PlanStep(BaseModel):
    agent_name: str
    agent_input: str


class Plan(BaseModel):
    steps: list[PlanStep]


class ExecuteEvent(Event):
    plan: Plan
    chat_history: list[ChatMessage]


class PlannerWorkflow(Workflow):
    llm: OpenAI = OpenAI(
        model="o3-mini",
        api_key="sk-proj-...",
    )
    agents: dict[str, FunctionAgent] = {
        "ResearchAgent": research_agent,
        "WriteAgent": write_agent,
        "ReviewAgent": review_agent,
    }

    @step
    async def plan(
        self, ctx: Context, ev: InputEvent
    ) -> ExecuteEvent | OutputEvent:
        # Set initial state if it exists
        if ev.state:
            await ctx.store.set("state", ev.state)

        chat_history = ev.chat_history

        if ev.user_msg:
            user_msg = ChatMessage(
                role="user",
                content=ev.user_msg,
            )
            chat_history.append(user_msg)

        # Inject the system prompt with state and available agents
        state = await ctx.store.get("state")
        available_agents_str = "\n".join(
            [
                f'<agent name="{agent.name}">{agent.description}</agent>'
                for agent in self.agents.values()
            ]
        )
        system_prompt = ChatMessage(
            role="system",
            content=PLANNER_PROMPT.format(
                state=str(state),
                available_agents=available_agents_str,
            ),
        )

        # Stream the response from the llm
        response = await self.llm.astream_chat(
            messages=[system_prompt] + chat_history,
        )
        full_response = ""
        async for chunk in response:
            full_response += chunk.delta or ""
            if chunk.delta:
                ctx.write_event_to_stream(
                    StreamEvent(delta=chunk.delta),
                )

        # Parse the response into a plan and decide whether to execute or output
        xml_match = re.search(r"(<plan>.*</plan>)", full_response, re.DOTALL)

        if not xml_match:
            chat_history.append(
                ChatMessage(
                    role="assistant",
                    content=full_response,
                )
            )
            return OutputEvent(
                response=full_response,
                chat_history=chat_history,
                state=state,
            )
        else:
            xml_str = xml_match.group(1)
            root = ET.fromstring(xml_str)
            plan = Plan(steps=[])
            for step in root.findall("step"):
                plan.steps.append(
                    PlanStep(
                        agent_name=step.attrib["agent"],
                        agent_input=step.text.strip() if step.text else "",
                    )
                )

            return ExecuteEvent(plan=plan, chat_history=chat_history)

    @step
    async def execute(self, ctx: Context, ev: ExecuteEvent) -> InputEvent:
        chat_history = ev.chat_history
        plan = ev.plan

        for step in plan.steps:
            agent = self.agents[step.agent_name]
            agent_input = step.agent_input
            ctx.write_event_to_stream(
                PlanEvent(
                    step_info=f'<step agent="{step.agent_name}">{step.agent_input}</step>'
                ),
            )

            if step.agent_name == "ResearchAgent":
                await call_research_agent(ctx, agent_input)
            elif step.agent_name == "WriteAgent":
                # Note: we aren't passing the input from the plan since
                # we're using the state to drive the write agent
                await call_write_agent(ctx)
            elif step.agent_name == "ReviewAgent":
                await call_review_agent(ctx)

        state = await ctx.store.get("state")
        chat_history.append(
            ChatMessage(
                role="user",
                content=f"I've completed the previous steps, here's the updated state:\n\n<state>\n{state}\n</state>\n\nDo you need to continue and plan more steps?, If not, write a final response.",
            )
        )

        return InputEvent(
            chat_history=chat_history,
        )