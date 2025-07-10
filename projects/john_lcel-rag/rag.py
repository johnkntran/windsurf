from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from store import retriever

prompt_template = ChatPromptTemplate([
    ("system", "Use the following context to answer the user's question: {context}"),
    ("user", "{question}")
])

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# LCEL chain: combines retrieval, context formatting, and model invocation
def format_docs(docs):
    return " ".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | model
    | StrOutputParser()
)

question = "What does Mark Carney say about President Trump?"

response = chain.invoke(question)

print(response)