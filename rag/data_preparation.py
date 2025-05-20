import bs4
import os 
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



# curl -fsSL https://ollama.com/install.sh | sh
# https://github.com/ollama/ollama

def resource_book( resource_path : str ) :
    loader  = PyPDFLoader(resource_path)
    docs    = loader.load()
    print(f"Docs: done")
    return docs

def text_splitting( docs ) :
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True )
    all_splits    = text_splitter.split_documents(docs)
    if Test: 
        vector_1 = embeddings.embed_query(all_splits[0].page_content)
        vector_2 = embeddings.embed_query(all_splits[1].page_content)
        assert len(vector_1) == len(vector_2)
        print(f"Generated vectors of length {len(vector_1)}\n")
        print(len(all_splits))
    return all_splits 

def storing( chunks ) :
    embeddings   = OllamaEmbeddings(model="llama3.2")
    vector_store = Chroma(
                collection_name    = "Linux_collection",
                embedding_function = embeddings,
                persist_directory  = "./chroma_langchain_db",  
                )

    ids = vector_store.add_documents(documents = chunks)
    return True if ids else False 


if __name__ == '__main__':

    Test= False
    user_input = input("Enter PDF Path : ")
    chunks     = text_splitting(resource_book(user_input))
    if storing(chunks) :
        print("PDF successfully stored in Vector DB Chroma")
    else : 
        print("FAILED storing PDF file")











