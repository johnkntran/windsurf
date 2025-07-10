from redis import Redis
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from redisvl.query.filter import Tag
import os


redis = Redis.from_url(os.environ['REDIS_URL'])
v = store = vector_store = RedisVectorStore(
    index_name="langchain-demo",
    embeddings=OpenAIEmbeddings(),
    redis_client=redis,
)

document_1 = Document(page_content="foo", metadata={"baz": "bar"})
document_2 = Document(page_content="bar", metadata={"foo": "baz"})
document_3 = Document(page_content="to be deleted")
documents = [document_1, document_2, document_3]
ids = ["1", "2", "3"]

store.add_documents(documents=documents, ids=ids)
store.delete(ids=["3"])

results = store.similarity_search_with_score(query="foo", k=1)
for doc, score in results:
    print(f"* [SIM={score:.3f}] content={doc.page_content} metadata={doc.metadata}")

docs = store.get_by_ids([1, 2])

retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
)
docs = retriever.get_relevant_documents("foo")

# ---------------------------------------------------------------------- #

from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine
import os


vector_store = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="langchain-demo",
    connection=os.environ['DATABASE_URL'],
    use_jsonb=True,
)

document_1 = Document(page_content="foo", metadata={"baz": "bar"})
document_2 = Document(page_content="thud", metadata={"bar": "baz"})
document_3 = Document(page_content="i will be deleted :(")

documents = [document_1, document_2, document_3]
ids = ["1", "2", "3"]

vector_store.add_documents(documents=documents, ids=ids)
vector_store.delete(ids=["3"])

results = vector_store.similarity_search(query="thud",k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

results = vector_store.similarity_search_with_score(query="qux",k=1)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
)
retriever.invoke("thud")

engine = create_async_engine(os.environ['DATABASE_URL'])

async_vector_store = PGVector(
    embeddings=OpenAIEmbeddings(),
    collection_name="langchain-demo",
    connection=engine,
    use_jsonb=True,
)

async def aget_query():
    results = await async_vector_store.asimilarity_search_with_score(query="qux",k=1)
    for doc,score in results:
        print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")