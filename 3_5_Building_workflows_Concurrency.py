import asyncio
import random
from llama_index.core.workflow import Workflow, Context, StartEvent, StopEvent, \
    step, Event


class StepTwoEvent(Event):
    query: str

class StepThreeEvent(Event):
    result: str


class ParallelFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StepTwoEvent | None:
        ctx.send_event(StepTwoEvent(query="Query 1"))
        ctx.send_event(StepTwoEvent(query="Query 2"))
        ctx.send_event(StepTwoEvent(query="Query 3"))

    @step(num_workers=4)
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        wait = random.randint(1, 5)
        print(f"Running slow query {ev.query} for {wait} secs")
        await asyncio.sleep(wait)
        return StopEvent(result=ev.query)


class ConcurrentFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StepTwoEvent | None:
        ctx.send_event(StepTwoEvent(query="Query 1"))
        ctx.send_event(StepTwoEvent(query="Query 2"))
        ctx.send_event(StepTwoEvent(query="Query 3"))

    @step(num_workers=4)
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StepThreeEvent:
        wait = random.randint(1, 5)
        print(f"Running slow query {ev.query} for {wait} secs")
        await asyncio.sleep(wait)
        return StepThreeEvent(result=ev.query)

    @step
    async def step_three(self, ctx: Context, ev: StepThreeEvent) -> StopEvent | None:
        # wait until we receive 3 events
        result = ctx.collect_events(ev, [StepThreeEvent] * 3)
        if result is None:
            return None

        # do something with all 3 results together
        print(result)
        return StopEvent(result="Done")


async def main():
    parellel_flow = ParallelFlow(timeout=10, verbose=False)
    concurrent_flow = ConcurrentFlow(timeout=10, verbose=False)
    print(await parellel_flow.run(first_input="Start the parellel workflow."))
    print(await concurrent_flow.run(first_input="Start the concurrent workflow."))


if __name__ == "__main__":
    asyncio.run(main())
