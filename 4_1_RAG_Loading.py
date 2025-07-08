from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

# ---------------------------------------------------------------------------- #

from llama_index.core import download_loader

from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    scheme=os.getenv("DB_SCHEME"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    dbname=os.getenv("DB_NAME"),
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)

# ---------------------------------------------------------------------------- #

from llama_index.core import Document

doc = Document(text="text")

# ---------------------------------------------------------------------------- #

from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents)
vector_index.as_query_engine()

# ---------------------------------------------------------------------------- #

from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(
    chunk_size=50,
    chunk_overlap=10,
    paragraph_separator='\n\n',
)

# global
from llama_index.core import Settings

Settings.text_splitter = text_splitter

# per-index
index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter]
)

retriever = index.as_retriever()
retriever.similarity_top_k = 3
nodes = retriever.retrieve('Donald Trump')

for node in nodes:
    print(node.text)
    print('----------')

engine = index.as_query_engine()
response = engine.query('What did Mark Carney say about Donald Trump?')
print(response)

# ---------------------------------------------------------------------------- #

from llama_index.core.schema import TextNode

node1 = TextNode(text="<text_chunk>", id_="<node_id>")
node2 = TextNode(text="<text_chunk>", id_="<node_id>")

index = VectorStoreIndex([node1, node2])

# ---------------------------------------------------------------------------- #

# writes all the data to disk
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")

# load index
index = load_index_from_storage(storage_context)

# ---------------------------------------------------------------------------- #

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma.db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# either A: load some documents and create new index from documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context
)

# or B: load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
)

# create a query engine and query
query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
print(response)

# ---------------------------------------------------------------------------- #

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)

# ---------------------------------------------------------------------------- #

query_engine = index.as_query_engine()
response = query_engine.query(
    "Write an email to the user given their background information."
)
print(response)

# ---------------------------------------------------------------------------- #

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# build index
index = VectorStoreIndex.from_documents(documents)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
response = query_engine.query("What did the author do growing up?")
print(response)