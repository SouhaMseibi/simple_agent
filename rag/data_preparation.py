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

load_dotenv() 

# LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
# LANGSMITH_ENDPOINT =  os.getenv("LANGSMITH_ENDPOINT")
# LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
# LANGSMITH_PROJECT = os.getenv("LANGSMITH_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Test = False 

# print('---------------------PDF_LOADER-------------------------------')

# file_path = "/home/souha/llm/linux_ch02.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()

# #Test : 
# # if Test : 
# #     print(f"{docs[0].page_content[:200]}\n")
# #     print('-----------------------------------------------------')
# #     print(docs[0].metadata)
# #     print('-----------------------------------------------------')
# #     print(docs[0].id)

# print('---------------------SPLITTER--------------------------------')

# #Define the chunker algorithm 
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(docs)

# # if Test : 
# #     print(len(all_splits))

# print('---------------------EMBEDDING--------------------------------')

# #Define the embedding model 
# embeddings = OllamaEmbeddings(model="llama3.2")

# # if Test: 
# #     vector_1 = embeddings.embed_query(all_splits[0].page_content)
# #     vector_2 = embeddings.embed_query(all_splits[1].page_content)
# #     assert len(vector_1) == len(vector_2)
# #     print(f"Generated vectors of length {len(vector_1)}\n")
# #     print(vector_1[:10])

# print('----------------------STORING-------------------------------')

# #Define the vector DB 
# vector_store = Chroma(
#     collection_name="Linux_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  
# )
# print("hggg")
# ids = vector_store.add_documents(documents=all_splits)
# print( f'Done storing to vector database {ids}')





def resource_book( resource_path : str ) :

    loader = PyPDFLoader(resource_path)

    docs = loader.load()
    print(f"Docs: done")
    return docs

def text_splitting( docs ) :

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True )
    all_splits    = text_splitter.split_documents(docs)
    print(len(all_splits))
    return all_splits 

def storing( chunks ) :

    embeddings = OllamaEmbeddings(model="llama3.2")
    vector_store = Chroma(
                collection_name    = "Linux_collection",
                embedding_function = embeddings,
                persist_directory  = "./chroma_langchain_db",  
                )

    ids = vector_store.add_documents(documents = chunks)

    return True if ids else False 


if __name__ == '__main__':

    user_input = input("Enter PDF Path : ")
    chunks = text_splitting(resource_book(user_input))
    if storing(chunks) :
        print("PDF successfully stored in Vector DB Chroma")
    else : 
        print("FAILED storing PDF file")











###EMBEDDING OPENAIAPI 
# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large" , api_key= OPENAI_API_KEY)

###Error code: 429 : insufficient quota
