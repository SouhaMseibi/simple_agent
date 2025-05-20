import bs4
import os 
import asyncio
from langchain import hub
from dotenv import load_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import chain
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from IPython.display import Image, display
from langchain_ollama import ChatOllama


load_dotenv() 


embeddings = OllamaEmbeddings(model="llama3.2")

vector_store = Chroma(
    collection_name="Linux_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  
   
)

prompt = hub.pull("rlm/rag-prompt")

Ollama_BASE_URL = os.getenv('Ollama_BASE_URL')

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    base_url=Ollama_BASE_URL
    
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"question": user_input}):
        if "generate" in event:
            print("Answer:", event["generate"]["answer"])
        else:
            print("Processing..." )



 
if __name__ == "__main__":
    while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)


