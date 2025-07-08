from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator

# create llm
llm = OpenAI(model="gpt-4", temperature=0.0)

# build index
...
vector_index = VectorStoreIndex(...)

# define evaluator
evaluator = FaithfulnessEvaluator(llm=llm)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))

# ---------------------------------------------------------------------------- #

from llama_index.core.evaluation import RetrieverEvaluator

# define retriever somewhere (e.g. from index)
# retriever = index.as_retriever(similarity_top_k=2)
retriever = ...

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

retriever_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)

# ---------------------------------------------------------------------------- #

from llama_index.core.llms import MockLLM
from llama_index.core import MockEmbedding, Settings

# use a mock llm globally
Settings.llm = MockLLM(max_tokens=256)

# use a mock embedding globally
Settings.embed_model = MockEmbedding(embed_dim=1536)

# ---------------------------------------------------------------------------- #

from llama_index.core.llms import MockLLM
from llama_index.core import MockEmbedding
from llama_index.core import Settings

llm = MockLLM(max_tokens=256)
embed_model = MockEmbedding(embed_dim=1536)

import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])

Settings.llm = llm
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(
    "./docs/examples/data/paul_graham"
).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("query")

print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)