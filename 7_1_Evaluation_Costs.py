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