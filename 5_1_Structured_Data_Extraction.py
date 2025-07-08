from datetime import datetime
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A line item in an invoice."""
    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""
    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )

# ---------------------------------------------------------------------------- #

from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from pathlib import Path
from pprint import pprint
import json

pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path("./data/giant_receipt.pdf"))
text = documents[0].text

llm = OpenAI(model="gpt-4o-mini")
sllm = llm.as_structured_llm(Invoice)

response = sllm.complete(text)
json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))
pprint(response.raw)

# ---------------------------------------------------------------------------- #

from llama_index.core.prompts import PromptTemplate

prompt = PromptTemplate(
    "Extract an invoice from the following text. "
    "If you cannot find an invoice ID, use the company name "
    "'{company_name}' and the date as the invoice ID: {text}"
)

response = llm.structured_predict(
    Invoice, prompt, text=text, company_name="Uber"
)
json_output = response.model_dump_json()
print(json.dumps(json.loads(json_output), indent=2))

# ---------------------------------------------------------------------------- #

from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(Invoice)

resp = llm.chat_with_tools(
    [tool],
    # chat_history=chat_history,  # can optionally pass in chat history instead of user_msg
    user_msg="Extract an invoice from the following text: " + text,
    tool_required=True,  # can optionally force the tool call
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_calls=False
)

outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "Invoice":
        outputs.append(Invoice(**tool_call.tool_kwargs))

# use your outputs
print(outputs[0])

# ---------------------------------------------------------------------------- #

from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(LineItem)

resp = llm.chat_with_tools(
    [tool],
    user_msg="Extract line items from the following text: " + text,
    allow_parallel_tool_calls=True,
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_calls=False
)

outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "LineItem":
        outputs.append(LineItem(**tool_call.tool_kwargs))

# use your outputs
print(outputs)

# ---------------------------------------------------------------------------- #

schema = Invoice.model_json_schema()
prompt = f"""
    Here is a JSON schema for an invoice: {json.dumps(schema)}

    Extract an invoice from the following text.
    Format your output as a JSON object according to the schema above.
    Do not include any other text than the JSON object.
    Omit any markdown formatting. Do not include any preamble or explanation.

    {text}
"""

response = llm.complete(prompt)

print(response)

invoice = Invoice.model_validate_json(response.text)

pprint(invoice)