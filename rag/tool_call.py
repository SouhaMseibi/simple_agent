import bs4
import os 
import re
from dotenv import load_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langchain.tools import  StructuredTool
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.tools import BraveSearch
from langchain_tavily import TavilySearch
import ast 

load_dotenv() 
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY') 



embeddings = OllamaEmbeddings(model="llama3.2")


vector_store = Chroma(
    collection_name   = "Linux_collection",
    embedding_function= embeddings,
    persist_directory = "./chroma_langchain_db",  
)

def get_prompt_template_prompty(category, name):
    return f"This is a {name} tool in the {category} category."


class State(TypedDict):
    question: str
    context: List[Document]
    link:str
    browsing_context:List[str]
    answer: str


@tool(description="Retrieve relevant documents from the vector store based on the question.")
def retrieve( state : State):

    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}



@tool(description="Load and extract content from web articles at the specified link.")
def browse_web_articles( state: State):

    search_tool = TavilySearch(
        max_results = 3,
        topic="general"
    )
    search_list=[]
    search_results = search_tool.invoke(state["link"])

    for i in range(len(search_results['results'])) :
        search_list.append(search_results['results'][i]['content'])

    return {"browsing_context": search_results if search_results else ["No content found"]}


#ToolNode will execute both tools in parallel
def tools_call(state: State):
    
    tool_node = ToolNode([retrieve, browse_web_articles])
    message_with_multiple_tool_calls = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve",
                        # "args": {"state": {"question": state["question"]}},
                        "args": {"state": state},
                        "id": "tool_call_id_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "browse_web_articles",
                        # "args": {"state": {"link": state["link"]}},
                        "args": {"state": state},
                        "id": "tool_call_id_2",
                        "type": "tool_call",
                    },
                ],
            )

    result = tool_node.invoke({"messages": [message_with_multiple_tool_calls]}) 
    return result 


if __name__ == "__main__":
    
    state = State(
        question="How do Linux permissions work?",
        context=[],
        link="https://www.redhat.com/en/blog/linux-file-permissions-explained",
        browsing_context=[],
        answer=""
    )
    
    # Call the tools
    result = tools_call(state)
    if result : 
        print("Tools successfully inovked \n")

        first_content = result['messages'][0].page_content
        second_content = result['messages'][1].content

        print(f" Resource retrieval : {first_content} \n")
        print(f" Web Search: {second_content} \n")

    else : 
        print("FAILED to invoke tools")





