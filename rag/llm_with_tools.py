import os
from langchain import hub
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, END, StateGraph
from typing_extensions import List, TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langchain.tools import  StructuredTool
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama


load_dotenv()


embeddings          = OllamaEmbeddings(model="llama3.2")
Ollama_BASE_URL     = os.getenv('Ollama_BASE_URL')
TAVILY_API_KEY      = os.getenv('TAVILY_API_KEY')

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    base_url=Ollama_BASE_URL

)

vector_store = Chroma(
    collection_name   = "Linux_collection",
    embedding_function= embeddings,
    persist_directory = "./chroma_langchain_db",  
)

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    browsing_context:List[str]
    answer: str

def retrieve(state: State):

    retrieved_docs = vector_store.similarity_search(state['question'])
    # if retrieved_docs :
        # print('Retrieved \n')
    return {"context": retrieved_docs if retrieved_docs else [] }


def browse_web_articles(state: State):

    search_tool = TavilySearch(
        max_results = 3,
        topic="general"
    )
    search_list=[]
    search_results = search_tool.invoke(state['question'])
    if search_results:
        # print('Web searched \n')
        for i in range(len(search_results['results'])) :
            search_list.append(search_results['results'][i]['content'])

    return {"browsing_context": search_results if search_results else ["No content found"]}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"],
         "context": docs_content ,
         "browsing_context": state["browsing_context"] })
    # print('In generate \n')
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("browse_web_articles", browse_web_articles)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "browse_web_articles")
graph_builder.add_edge("browse_web_articles", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input):

    for event in graph.stream({"question": user_input}):

        if "generate" in event:
            print("Answer:", event["generate"]["answer"])

while True:

        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input )