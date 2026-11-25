import os
import sys
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

# Initialize FastAPI
app = FastAPI(
    title="Python LLM Pipeline API",
    description="Serverless LLM Pipeline with Groq & OpenAI on Vercel",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_client_and_model():
    """Retrieve credentials from Vercel environment."""
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback for local testing if .env exists
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set in environment variables. Please add it in Vercel Project Settings > Environment Variables."
        )

    api_key = api_key.strip()
    if api_key.startswith("gsk_"):
        base_url = "https://api.groq.com/openai/v1"
        model = os.getenv("CHAT_MODEL") or "openai/gpt-oss-120b"
    else:
        base_url = "https://api.openai.com/v1"
        model = os.getenv("CHAT_MODEL") or "gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model

# Schemas
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are an expert Python coding instructor. Provide concise, high quality responses with code examples."
    temperature: Optional[float] = 0.7

class TutorialStep(BaseModel):
    step_number: int = Field(description="Sequential order of step")
    title: str = Field(description="Step title")
    code: str = Field(description="Python code snippet")
    explanation: str = Field(description="Explanation of the step")

class TutorialArticle(BaseModel):
    title: str
    summary: str
    difficulty: str
    prerequisites: List[str]
    steps: List[TutorialStep]

class TutorialRequest(BaseModel):
    topic: str
    difficulty: Optional[str] = "Beginner"

class RAGRequest(BaseModel):
    query: str

# In-memory document storage for serverless RAG
DEFAULT_KNOWLEDGE = [
    "LlamaIndex is a powerful data framework connecting custom data sources to Large Language Models (LLMs).",
    "Groq provides ultra-low latency LPU (Language Processing Unit) inference for open-source AI models.",
    "FastAPI is a modern, high-performance web framework for building APIs with Python based on standard type hints.",
    "Python generators use the yield keyword to produce values lazily on demand, saving memory.",
    "Vercel is a cloud platform for frontend frameworks and serverless functions, supporting instantaneous edge deployments.",
]

@app.get("/api/health")
def health():
    client, model = get_client_and_model()
    return {"status": "ok", "model": model}

@app.post("/api/chat")
def chat(req: ChatRequest):
    client, model = get_client_and_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.prompt},
            ],
            temperature=req.temperature,
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    client, model = get_client_and_model()
    try:
        def generate():
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": req.system_prompt},
                    {"role": "user", "content": req.prompt},
                ],
                temperature=req.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {delta}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tutorial", response_model=TutorialArticle)
def generate_tutorial(req: TutorialRequest):
    client, model = get_client_and_model()
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior Python curriculum developer creating clean, structured tutorials.",
                },
                {
                    "role": "user",
                    "content": f"Create a {req.difficulty}-level Python tutorial about: {req.topic}",
                },
            ],
            response_format=TutorialArticle,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag")
def rag_search(req: RAGRequest):
    """Serverless RAG using contextual passage matching."""
    client, model = get_client_and_model()
    try:
        # Check local data files if available
        context_docs = list(DEFAULT_KNOWLEDGE)
        data_path = Path(__file__).resolve().parent.parent / "data"
        if data_path.exists():
            for f in data_path.glob("*.txt"):
                try:
                    text = f.read_text(encoding="utf-8").strip()
                    if text:
                        context_docs.append(text)
                except Exception:
                    pass

        joined_context = "\n- ".join(context_docs)
        prompt = (
            f"Context information:\n- {joined_context}\n\n"
            f"Based strictly on the context above, answer this question: {req.query}\n"
            f"If the answer cannot be found in the context, state that clearly."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an accurate RAG question answering assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return {"query": req.query, "answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
