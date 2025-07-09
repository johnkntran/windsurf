from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ---------------------------------------------------------------------------- #

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.redis import RedisVectorStore

# load documents
documents = SimpleDirectoryReader(
    input_files=["./data/barack-obama-a-more-perfect-union.txt"]
).load_data()

# ---------------------------------------------------------------------------- #

from llama_index.core import StorageContext
from redis import Redis
import os

# create a Redis client connection
redis_url = os.environ['REDIS_URL']
redis_client = Redis.from_url(redis_url)
redis_client.ping()

# create the vector store wrapper
vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)

# load storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# A: build and load index from documents and storage context
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
)

# B: load an existing index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# ---------------------------------------------------------------------------- #

import textwrap

query_engine = index.as_query_engine(similarity_top_k=7)
retriever = index.as_retriever(similarity_top_k=7)

result_nodes = retriever.retrieve("What did Barack Obama say about Reverend Wright?")
for node in result_nodes:
    print(node)
    print('----------')

response = query_engine.query("What did Barack Obama say about Reverend Wright?")
print(textwrap.fill(str(response), 100))

# ---------------------------------------------------------------------------- #

index.vector_store.delete_index()