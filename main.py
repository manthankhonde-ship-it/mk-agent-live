from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph_brain import app as ai_graph 
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = ai_graph.invoke({"question": request.question})
        return {"answer": result.get("answer", "Error occurred")}
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}"}

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")