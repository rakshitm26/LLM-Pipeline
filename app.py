"""
FastAPI Production Server for LLM Pipeline.
Exposes endpoints for Chat, Streaming, Structured Tutorial Generation, and RAG.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pipeline import LLMPipeline, TutorialArticle
from config import handle_openai_error

app = FastAPI(
    title="LLM Pipeline API",
    description="Production-ready REST API for LLM chat, RAG, and structured generation.",
    version="1.0.0",
)

# Enable CORS for frontend integration (React, Vue, mobile, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LLMPipeline()

# Request schemas
class ChatRequest(BaseModel):
    prompt: str = Field(..., example="What is a Python generator?")
    system_prompt: Optional[str] = Field(
        default="You are a helpful Python expert.",
        example="You are a helpful assistant.",
    )

class TutorialRequest(BaseModel):
    topic: str = Field(..., example="FastAPI Dependency Injection")
    difficulty: Optional[str] = Field(default="Beginner", example="Beginner")

class RAGRequest(BaseModel):
    query: str = Field(..., example="What is LlamaIndex?")

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Pipeline API"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Standard non-streaming chat completion."""
    try:
        response = pipeline.chat(prompt=req.prompt, system_prompt=req.system_prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    """Real-time token streaming using Server-Sent Events (SSE)."""
    try:
        def event_stream():
            for chunk in pipeline.stream_chat(prompt=req.prompt, system_prompt=req.system_prompt):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tutorial", response_model=TutorialArticle)
def generate_tutorial_endpoint(req: TutorialRequest):
    """Generate structured, validated Python tutorials."""
    try:
        return pipeline.generate_tutorial(topic=req.topic, difficulty=req.difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
def rag_endpoint(req: RAGRequest):
    """Query ingested documents from ./data using RAG."""
    try:
        answer = pipeline.rag_query(req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
