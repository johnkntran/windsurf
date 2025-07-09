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

# create the vector store wrapper
vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)

# load storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# build and load index from documents and storage context
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)