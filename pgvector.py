from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
import textwrap
import openai
import psycopg2
import os


openai.api_key = os.environ["OPENAI_API_KEY"]

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
connection_string = os.environ["DATABASE_URL"]
db_name = "llamaindex"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

# ---------------------------------------------------------------------------- #

from sqlalchemy import make_url

url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay",
    embed_dim=1536,  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# A: For indexing new documents
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    show_progress=True,
)

# B: For using existing index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine()

# ---------------------------------------------------------------------------- #

response = query_engine.query("What did the author do?")
print(textwrap.fill(str(response), 100))

# ---------------------------------------------------------------------------- #

from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="author", value="mats@timescale.com"),
        MetadataFilter(key="author", value="sven@timescale.com"),
    ],
    condition="or",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    print(node.node.metadata)