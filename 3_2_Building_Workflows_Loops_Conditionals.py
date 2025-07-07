# Branches & Loops #

import os
import random
import asyncio
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)

class FirstEvent(Event):
    first_output: str

class SecondEvent(Event):
    second_output: str

class LoopEvent(Event):
    loop_output: str

class BranchA(Event):
    payload: str

class BranchB(Event):
    payload: str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if isinstance(ev, StartEvent):
            print(ev.first_input)
        elif isinstance(ev, LoopEvent):
            print(ev.loop_output)
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> BranchA | BranchB:
        print(ev.first_output)
        if random.randint(0, 1) == 0:
            return BranchA(payload='Second step complete on Branch A')
        else:
            return BranchB(payload='Second step complete on Branch B')

    @step
    async def step_three(self, ev: BranchA | BranchB) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Workflow complete.")

async def main():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run(first_input="Start the workflow.")
    print(result)
    if os.getenv("DRAW_WORKFLOW"):
        from llama_index.utils.workflow import draw_all_possible_flows
        draw_all_possible_flows(workflow, filename="basic_workflow.html")

if __name__ == "__main__":
    asyncio.run(main())
