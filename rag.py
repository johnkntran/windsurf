from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from store import retriever


prompt_template = ChatPromptTemplate([
    ("system", "Use the following context to answer the user's question: {context}"),
    ("user", "{question}")
])

question = "What does Mark Carney say about President Trump?"

docs = retriever.invoke(question)
context = " ".join(d.page_content for d in docs)

messages = prompt_template.invoke({"context": context, "question": question})

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

response = model.invoke(messages)

print(response.content)