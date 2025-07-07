import asyncio
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)

class SetupEvent(Event):
    query: str


class StepTwoEvent(Event):
    query: str


class StatefulFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> SetupEvent | StepTwoEvent:
        db = await ctx.store.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)
        # do something with the query
        return StepTwoEvent(query=ev.query)

    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        # load data
        await ctx.store.set("some_database", [1, 2, 3])
        return StartEvent(query=ev.query)

    @step
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        # do something with the data
        print("Data is ", await ctx.store.get("some_database"))
        return StopEvent(result=await ctx.store.get("some_database"))

async def main():
    w = StatefulFlow(timeout=10, verbose=False)
    result = await w.run(query="Some query")
    print(result)

if __name__ == '__main__':
    asyncio.run(main())

# Pydantic example #