from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import rag.rag 
from rag.rag import stream_graph_updates , graph
import os
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from IPython.display import Image, display
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(model="llama3.2")



vector_store = Chroma(
    collection_name="Linux_collection",
    embedding_function=embeddings,
    # persist_directory="./chroma_langchain_db",  
    persist_directory="/home/souha/llm/chroma_langchain_db",
)

app = FastAPI()


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            #chat-container {
                display: flex;
                flex-direction: column;
                height: 400px;
                border: 1px solid #ccc;
                border-radius: 5px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 10px;
            }
            .message {
                padding: 8px 12px;
                margin-bottom: 10px;
                border-radius: 5px;
                max-width: 70%;
            }
            .user-message {
                background-color: #e9f5ff;
                align-self: flex-end;
                margin-left: auto;
            }
            .ai-message {
                background-color: #f1f1f1;
                align-self: flex-start;
            }
            #message-form {
                display: flex;
            }
            #message-input {
                flex-grow: 1;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-right: 10px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>RAG Chat Interface</h1>
        <div id="chat-container"></div>
        <form id="message-form" onsubmit="sendMessage(event)">
            <input type="text" id="message-input" placeholder="Type your message..." autocomplete="off"/>
            <button type="submit">Send</button>
        </form>
        <script>
            const chatContainer = document.getElementById('chat-container');
            // const ws = new WebSocket("ws://localhost:8000/ws"); 
            ws = new WebSocket("ws://" + window.location.host + "/ws");
            
            
            // When the connection is established
            ws.onopen = function(event) {
                console.log("Connection established");
            };
            
            // When we receive a message
            ws.onmessage = function(event) {
                const message = document.createElement('div');
                message.className = 'message ai-message';
                message.textContent = event.data;
                chatContainer.appendChild(message);
                // Auto scroll to bottom
                chatContainer.scrollTop = chatContainer.scrollHeight;
            };
            
            // When there's an error
            ws.onerror = function(event) {
                console.error("WebSocket error:", event);
            };
            
            // Function to send a message
            function sendMessage(event) {
                event.preventDefault();
                const input = document.getElementById("message-input");
                const message = input.value.trim();
                
                if (message) {
                    // Add user message to chat
                    const userMessage = document.createElement('div');
                    userMessage.className = 'message user-message';
                    userMessage.textContent = message;
                    chatContainer.appendChild(userMessage);
                    
                    // Send message to server
                    ws.send(message);
                    
                    // Clear input
                    input.value = '';
                    
                    // Auto scroll to bottom
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_input = await websocket.receive_text()
            print(f"Received: {user_input}")
            
            # Stream responses from the LangGraph chain
            async for update in stream_graph_updates(user_input):
                # Send the actual update content instead of wrapping it
                await websocket.send_text(update)
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.send_text(f"Error: {str(e)}")