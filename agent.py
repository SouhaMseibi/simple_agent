# from langchain_community.tools import BraveSearch
# from langchain_anthropic import ChatAnthropic
# from langgraph.prebuilt import create_react_agent
# from langchain.chat_models import init_chat_model
# from langchain_openai import OpenAI
# from langchain_core.messages import HumanMessage
# import os 
# from langchain_openai import OpenAI


# os.environ["OPENAI_API_KEY"]="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"
# llm = OpenAI()
# print(llm.invoke("Hello how are you?"))
# api_key = "BSAjg36PROzydteRRAfkVDn3b9y8OnY"

# search_tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
# search_results= search_tool.invoke("what is the weather in SF")

# print(search_results)


# model = init_chat_model("gpt-4", model_provider="openai")


import os 
import getpass
import time
from langchain_community.tools import BraveSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from openai import OpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.output_parsers import StrOutputParser


api_key = "BSAjg36PROzydteRRAfkVDn3b9y8OnY"
# os.environ["TAVILY_API_KEY"]="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"
# os.environ["OPENAI_API_KEY"]="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"


# model = init_chat_model("gpt-3.5-turbo", model_provider="openai").with_retry(stop_after_attempt=6)


template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)
search_tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
model = OllamaLLM(model="llama3.2" ,  base_url="http://localhost:11434")

chain = prompt | model 

response = chain.invoke({"question": "What is LangChain?"})
# search_results= search_tool.invoke("what is the weather in SF")

print(response)


#With Ollama :  bind not working ,  create_react_agent no bind-tools in Ollama
#Wicth OpenApi : Error code: 429 - {'error_message': 'You exceeded your current quota'}
#RAG Part 1 