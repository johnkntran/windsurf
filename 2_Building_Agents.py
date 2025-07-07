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