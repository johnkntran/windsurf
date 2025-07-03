from redis import Redis
from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from redisvl.query.filter import Tag


redis = Redis('cache')
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