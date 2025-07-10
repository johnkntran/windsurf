from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


connection = "postgresql+psycopg://langchain:langchain@database:5432/langchain"
collection_name = "mark_carney_acceptance_speech"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
engine = create_async_engine(connection)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

async_vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=engine,
    use_jsonb=True,
)

retriever = vector_store.as_retriever(
    # search_type="mmr",
    # search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
)


def load_docs():
    raw_documents = TextLoader('mark-carney-acceptance-speech.txt').load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    docs = text_splitter.split_documents(raw_documents)
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    documents = text_splitter.split_documents(raw_documents)
    ids = [*range(1, len(documents) + 1)]
    vector_store.add_documents(documents=documents, ids=ids)
    trump_docs = vector_store.get_by_ids(['14', '25', '40'])

def perform_search():
    results = vector_store.similarity_search_with_score(query="President Trump",k=5)
    for doc, score in results:
        print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

async def perform_async_search():
    results = await async_vector_store.asimilarity_search_with_score(query="President Trump",k=5)
    for doc, score in results:
        print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")