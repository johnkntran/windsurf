# 5-Line Starter #

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did Mark Carney say about President Trump?")
print(response)

# Using LLMs #

from llama_index.llms.openai import OpenAI

response = OpenAI().complete("William Shakespeare is ")
print(response)

handle = OpenAI().stream_complete("William Shakespeare is ")
for token in handle:
    print(token.delta, end="", flush=True)

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.chat(messages)

llm = OpenAI(model="gpt-4o-mini")
response = llm.complete("Who is Laurie Voss?")
print(response)

# Multi-modal LLMs #

from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="data/wallpaper.jpg"),
            TextBlock(text="Describe the image in a few sentences."),
        ],
    )
]

resp = llm.chat(messages)
print(resp.message.content)

# Tool Calling #

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from typing import TypedDict

class Song(TypedDict):
    name: str
    artist: str

def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)

tool = FunctionTool.from_defaults(fn=generate_song)

llm = OpenAI(model="gpt-4o")
response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
print(str(response))
