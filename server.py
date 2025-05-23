from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
import jwt

app = FastAPI()

# Enable CORS for React front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables (replace with your keys)
os.environ["OPENAI_API_KEY"] = "sk-67nIiA7-e2rxldl5qaOPRK-BvJ8O2lsJLhdeVtRZ_cT3BlbkFJSqLX3tdehY3tObgIyGgPOuIIAkn7kM4avkiO1vRTwA"
os.environ["TAVILY_API_KEY"] = "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"

# Supabase setup
supabase: Client = create_client("https://lakxetbfkozevwlelzaj.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxha3hldGJma296ZXZ3bGVsemFqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODAxNzI3MywiZXhwIjoyMDYzNTkzMjczfQ.lORYixYT1P4COjSGxkU_peC7PgJhni4wO91xinovnW8")

# OAuth2 for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Agent storage (in-memory for simplicity; use Supabase in production)
agents = {}
memories = {}

class AgentConfig(BaseModel):
    user_id: str
    name: str
    prompt: str
    tools: list[str]
    training_data: str

class ChatRequest(BaseModel):
    user_id: str
    agent_name: str
    message: str

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, "7cxZdJVNMggD/GpYnpljui3QmDlmJGkp7DsREjPDeY8+MDpEjMewc8qw7JLi+6d6nseAtBSBBNrTTPoPstyzZQ==", algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/create_agent")
async def create_agent(config: AgentConfig, current_user: str = Depends(get_current_user)):
    if current_user != config.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        tools = []
        if "web_search" in config.tools:
            tools.append(TavilySearchResults(max_results=3))
        
        # Handle training data
        if config.training_data:
            # Save training data to temporary file (replace with Supabase storage in production)
            with open(f"temp_{config.user_id}_{config.name}.txt", "w") as f:
                f.write(config.training_data)
            loader = TextLoader(f"temp_{config.user_id}_{config.name}.txt")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
            retriever = vectorstore.as_retriever()
            tools.append(retriever)
        
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "chat_history"],
            template=f"{config.prompt}\n\nChat History:\n{{chat_history}}\n\nUser Input: {{input}}\nAgent Scratchpad: {{agent_scratchpad}}"
        )
        
        agent = create_react_agent(llm, tools, prompt)
        memory = ConversationBufferMemory()
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
        
        # Store agent and memory
        agent_key = f"{config.user_id}_{config.name}"
        agents[agent_key] = agent_executor
        memories[agent_key] = memory
        
        # Save to Supabase
        supabase.table("agents").insert({
            "user_id": config.user_id,
            "name": config.name,
            "prompt": config.prompt,
            "tools": config.tools,
            "training_data": config.training_data
        }).execute()
        
        return {"status": "Agent created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    if current_user != request.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    agent_key = f"{request.user_id}_{request.agent_name}"
    agent_executor = agents.get(agent_key)
    if not agent_executor:
        raise HTTPException(status_code=400, detail="Agent not found")
    
    try:
        response = agent_executor.invoke({"input": request.message})
        # Save chat history to Supabase
        supabase.table("chat_history").insert({
            "user_id": request.user_id,
            "agent_name": request.agent_name,
            "message": request.message,
            "response": response["output"]
        }).execute()
        return {"response": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/widget/{user_id}/{agent_name}")
async def get_widget(user_id: str, agent_name: str):
    # Serve a simple chat widget
    return {
        "html": f"""
        <div id="chat-widget">
          <div id="messages" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;"></div>
          <input id="input" type="text" style="width: 100%; padding: 5px;" placeholder="Type your message...">
          <button onclick="sendMessage()">Send</button>
          <script>
            async function sendMessage() {{
              const input = document.getElementById('input').value;
              const messages = document.getElementById('messages');
              messages.innerHTML += `<div>User: ${input}</div>`;
              const response = await fetch('http://localhost:8000/chat', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ user_id: '{user_id}', agent_name: '{agent_name}', message: input }})
              }});
              const data = await response.json();
              messages.innerHTML += `<div>Agent: ${data.response}</div>`;
              document.getElementById('input').value = '';
            }}
          </script>
        </div>
        """
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)