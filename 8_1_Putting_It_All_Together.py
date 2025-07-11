# Semantic Search #

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

# Summarization #

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("<summarization_query>")

# Routing over Heterogeneous Data #

from llama_index.core import TreeIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine

# define sub-indices
index1 = VectorStoreIndex.from_documents(notion_docs)
index2 = VectorStoreIndex.from_documents(slack_docs)

# define query engines and tools
tool1 = QueryEngineTool.from_defaults(
    query_engine=index1.as_query_engine(),
    description="Use this query engine to do...",
)
tool2 = QueryEngineTool.from_defaults(
    query_engine=index2.as_query_engine(),
    description="Use this query engine for something else...",
)

query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2]
)

response = query_engine.query(
    "In Notion, give me a summary of the product roadmap."
)

# Compare/Contrast Queries #

from llama_index.core.query.query_transform.base import DecomposeQueryTransform

decompose_transform = DecomposeQueryTransform(
    service_context.llm, verbose=True
)

# Multi-Document Queries #

from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=sept_engine,
        name="sept_22",
        description="Provides information about Uber quarterly financials ending September 2022",
    ),
    QueryEngineTool.from_defaults(
        query_engine=june_engine,
        name="june_22",
        description="Provides information about Uber quarterly financials ending June 2022",
    ),
    QueryEngineTool.from_defaults(
        query_engine=march_engine,
        name="march_22",
        description="Provides information about Uber quarterly financials ending March 2022",
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

# Putting It All Together: Agent #

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import QueryEngineTool

# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

# initialize llm
llm = OpenAI(model="gpt-4o")

# initialize agent
agent = FunctionAgent(
    tools=[multiply],
    system_prompt="You are an agent that can invoke a tool for multiplication when assisting a user.",
)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=sql_agent,
        name="sql_agent",
        description="Agent that can execute SQL queries.",
    ),
]

agent = FunctionAgent(
    tools=query_engine_tools,
    system_prompt="You are an agent that can invoke an agent for text-to-SQL execution.",
)