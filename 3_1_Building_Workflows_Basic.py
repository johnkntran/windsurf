# A Basic Workflow #

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

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> FirstEvent:
        print(ev.first_input)
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")

async def main():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run(first_input="Start the workflow.")
    print(result)
    import os
    if os.getenv("DRAW_WORKFLOW"):
        from llama_index.utils.workflow import draw_all_possible_flows
        draw_all_possible_flows(workflow, filename="basic_workflow.html")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
