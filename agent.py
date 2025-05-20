
import os 
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_community.tools import BraveSearch
from langchain_core.prompts import  ChatPromptTemplate 
from langchain_core.tools import tool 
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.agents import create_openai_functions_agent,create_react_agent,create_structured_chat_agent,  AgentExecutor
import requests
from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOpenAI

load_dotenv() 
Ollama_BASE_URL = os.getenv('Ollama_BASE_URL')
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY') 


@tool
def search_tool(query: str) -> str:
    """Search the web using Brave Search."""
    search_tool = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 3})
    search_results = search_tool.invoke(query)
    return search_results

tools=[search_tool]


template = """\
You are an assistant to help answer questions . 
Questions : {input}
Context:{context}
"""

prompt = ChatPromptTemplate.from_template(  template) #from_messages

model = OllamaLLM(model="llama3.2" ,  base_url=Ollama_BASE_URL)

chain = prompt | model 

for token in chain.stream({"input": "what is LangChain?"}):
    print(token, end="", flush=True)





# system_prompt = """\

# You are an assistant that has access to tools {tools} to answer questions, use them when necessary.
# Action: the action to take, should be one of [{tool_names}]
# You can use the following tool to answer the question:

# Question: {input}
# {agent_scratchpad}
# """
# agent_prompt = ChatPromptTemplate.from_template(system_prompt)
# # agent_prompt = ChatPromptTemplate.from_messages([
# #     ("system", "You are a helpful assistant with access to the following tools: {tool_names},  {tools} . {agent_scratchpad}"),
# #     ("human", "{input}"),
    
# # ])
# agent = create_structured_chat_agent(model, tools, agent_prompt )  

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) #pass error back to the agent and have it try again

# print("Testing agent:")
# result = agent_executor.invoke({'input': "What is LangChain?"})
# print(result)








