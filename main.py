from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


connection = "postgresql+psycopg://langchain:langchain@database:5432/langchain"
collection_name = "mark_carney_acceptance_speech"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

raw_documents = TextLoader('mark-carney-acceptance-speech.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
docs = text_splitter.split_documents(raw_documents)
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=25)
documents = text_splitter.split_documents(raw_documents)
ids = [*range(1, len(documents) + 1)]

vector_store.add_documents(documents=documents, ids=ids)

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

engine = create_async_engine(connection)

async_vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=engine,
    use_jsonb=True,
)

results = await async_vector_store.asimilarity_search_with_score(query="qux",k=1)
for doc,score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")