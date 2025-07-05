from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from store import retriever

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Use the following context to answer the user's question: {context}"),
    ("user", "{question}")
])

# Define the model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the chain using LCEL
chain = (
    {
        "context": lambda x: " ".join(d.page_content for d in retriever.invoke(x["question"])),
        "question": lambda x: x["question"]
    }
    | prompt_template
    | model
)

# Example usage
question = "What does Mark Carney say about President Trump?"
response = chain.invoke({"question": question})
print(response.content)