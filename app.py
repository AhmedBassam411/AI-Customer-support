from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agents import support_bot
from fastapi import WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates

class ChatRequest(BaseModel):
    session_id: str
    message: str

app = FastAPI(title="AI Customer Support Chatbot")
templates = Jinja2Templates(directory="templates")

# Store active websocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.get("/")
async def get_chat_ui(request: Request):
    # Serve the frontend HTML
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    
    # LangGraph Thread Configuration
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        while True:
            # 1. Receive User Message
            user_text = await websocket.receive_text()
            
            # Send immediate typing status
            await websocket.send_text(json.dumps({"type": "status", "data": "Agent is typing..."}))
            
            # 2. Invoke LangGraph Multi-Agent Pipeline
            input_state = {"messages": [HumanMessage(content=user_text)]}
            output_state = support_bot.invoke(input_state, config=config)
            
            # 3. Extract final AI response & Escalation Flag
            ai_response = output_state["messages"][-1].content
            is_escalated = output_state.get("escalate", False)
            
            # 4. Push response back to frontend via WebSockets
            payload = {
                "type": "message",
                "data": ai_response,
                "escalated": is_escalated
            }
            await websocket.send_text(json.dumps(payload))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Session {session_id} disconnected.")
        from pydantic import BaseModel


@app.post("/api/chat")
def rest_chat_endpoint(request: ChatRequest):
    """REST endpoint specifically designed for the Streamlit frontend."""
    # Configure memory tracking via session_id
    config = {"configurable": {"thread_id": request.session_id}}
    
    # Pass user message into the LangGraph state machine
    input_state = {"messages": [HumanMessage(content=request.message)]}
    output_state = support_bot.invoke(input_state, config=config)
    
    # Extract AI response and escalation flag
    ai_response = output_state["messages"][-1].content
    is_escalated = output_state.get("escalate", False)
    
    return {
        "response": ai_response,
        "escalated": is_escalated
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.1", port=8000, reload=True)