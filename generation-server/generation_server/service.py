import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

if not os.environ.get("OPENAI_API_KEY"):
    exit("Please set OPENAI_API_KEY environment variable")

CHROMA_HOST = os.environ.get("CHROMA_HOST")
CHROMA_PORT = os.environ.get("CHROMA_PORT", "8000")
MODEL = os.environ.get("MODEL", "gpt-3o-mini")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")


load_dotenv()

model = MODEL

llm = ChatOpenAI(temperature=0.5, model=model, max_tokens=4096)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

print(f'connecting to vector DB {CHROMA_HOST}:{CHROMA_PORT}, in collection {COLLECTION_NAME}')
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
print(chroma_client.heartbeat())

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    client=chroma_client
)
retriever = vector_store.as_retriever()

template = """
**Context**: {context}
You are an expert in user manuals, your task is to provide accurate and useful information about the user manual. Use only the information provided in the documents to answer.
**User Question**: {question}

**Your Response**:
1. Provide a clear and structured answer.
2. Include examples, include prices, recommendations, or analogies if applicable.
3. Do not include any additional explanations or context.
4. Do not hallucinate.
5. Provide the page number for reference.
7. Answer the question in the language of the same question.
"""
prompt_template = ChatPromptTemplate([
    ("system", template),
    ("user", "{question}")
])

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"],k=3)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def generate_response(question: str):
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question": question})
    return response["answer"]