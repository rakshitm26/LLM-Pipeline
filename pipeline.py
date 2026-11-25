"""
Production-Ready LLM Pipeline (Groq & OpenAI Compatible).
Integrates Chat, Streaming, RAG over data/, and Structured Tutorial Generation.
Configured via .env backend.
"""

import os
import sys
from typing import Generator, List, Optional, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from fastembed import TextEmbedding
from llama_index.core.embeddings import BaseEmbedding
from config import get_credentials, get_openai_client, DEFAULT_CHAT_MODEL

class FastEmbedAdapter(BaseEmbedding):
    """Zero-cost local fast embedding adapter."""
    _model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _get_query_embedding(self, query: str) -> List[float]:
        return list(self._model.embed([query]))[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return list(self._model.embed([text]))[0].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

# Structured output schemas
class TutorialStep(BaseModel):
    step_number: int = Field(description="Order of step")
    title: str = Field(description="Short title")
    code: str = Field(description="Executable Python snippet")
    explanation: str = Field(description="Summary of purpose")

class TutorialArticle(BaseModel):
    title: str
    summary: str
    difficulty: str
    prerequisites: List[str]
    steps: List[TutorialStep]

class LLMPipeline:
    """Unified LLM Pipeline engine supporting Groq and OpenAI."""

    def __init__(self, data_dir: Optional[str] = None):
        self.api_key, self.base_url, self.model_name, self.provider = get_credentials()
        self.client: OpenAI = get_openai_client()
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self._rag_engine = None

    def chat(self, prompt: str, system_prompt: str = "You are a helpful Python expert.") -> str:
        """Send prompt and return text response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content or ""

    def stream_chat(self, prompt: str, system_prompt: str = "You are a helpful Python expert.") -> Generator[str, None, None]:
        """Stream response tokens one by one."""
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate_tutorial(self, topic: str, difficulty: str = "Beginner") -> TutorialArticle:
        """Generate a structured, step-by-step Python tutorial using Pydantic validation."""
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a senior Python instructor writing concise, practical guides."},
                {"role": "user", "content": f"Write a {difficulty}-level Python tutorial about: {topic}"},
            ],
            response_format=TutorialArticle,
        )
        return completion.choices[0].message.parsed

    def _init_rag(self):
        """Lazy load LlamaIndex RAG index."""
        if self._rag_engine is None:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

            # Configure LLM
            if self.provider == "groq":
                from llama_index.llms.groq import Groq
                Settings.llm = Groq(model=self.model_name, api_key=self.api_key)
            else:
                from llama_index.llms.openai import OpenAI as LlamaOpenAI
                Settings.llm = LlamaOpenAI(model=self.model_name, api_key=self.api_key)

            # Local zero-cost embeddings
            Settings.embed_model = FastEmbedAdapter()

            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir, exist_ok=True)

            documents = SimpleDirectoryReader(self.data_dir).load_data()
            if not documents:
                raise ValueError(f"No documents found in '{self.data_dir}' to build RAG index.")

            index = VectorStoreIndex.from_documents(documents)
            self._rag_engine = index.as_query_engine(similarity_top_k=2)

    def rag_query(self, query: str) -> str:
        """Query documents stored in the data directory using RAG."""
        self._init_rag()
        response = self._rag_engine.query(query)
        return str(response)


def interactive_cli():
    """Interactive command-line interface for the LLM pipeline."""
    pipeline = LLMPipeline()
    print("=" * 65)
    print(f"  🚀 LLM Pipeline ({pipeline.provider.upper()} - {pipeline.model_name})")
    print("=" * 65)
    print("1. Standard Chat")
    print("2. Streaming Chat")
    print("3. Generate Structured Python Tutorial")
    print("4. RAG Query (over docs in ./data)")
    print("5. Exit")
    print("-" * 65)

    while True:
        try:
            choice = input("\nSelect mode [1-5]: ").strip()
            if choice == "1":
                query = input("\nEnter your question: ").strip()
                if query:
                    print("\nResponse:")
                    print(pipeline.chat(query))
            elif choice == "2":
                query = input("\nEnter your question: ").strip()
                if query:
                    print("\nStreaming response: ", end="")
                    for chunk in pipeline.stream_chat(query):
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    print()
            elif choice == "3":
                topic = input("\nEnter tutorial topic (e.g. 'Decorators in Python'): ").strip()
                level = input("Difficulty [Beginner/Intermediate/Advanced] (default Beginner): ").strip() or "Beginner"
                if topic:
                    print(f"\nGenerating structured tutorial on '{topic}'...")
                    tut = pipeline.generate_tutorial(topic, level)
                    print(f"\n# {tut.title}")
                    print(f"**Difficulty:** {tut.difficulty}")
                    print(f"**Summary:** {tut.summary}")
                    print(f"**Prerequisites:** {', '.join(tut.prerequisites)}")
                    for step in tut.steps:
                        print(f"\n### Step {step.step_number}: {step.title}")
                        print(step.explanation)
                        print("```python")
                        print(step.code.strip())
                        print("```")
            elif choice == "4":
                query = input("\nQuery your documents in ./data: ").strip()
                if query:
                    print("\nRetrieving answer from RAG index...")
                    print(pipeline.rag_query(query))
            elif choice == "5" or choice.lower() in ("exit", "quit", "q"):
                print("Exiting pipeline. Happy coding!")
                break
            else:
                print("Invalid choice, please select 1-5.")
        except KeyboardInterrupt:
            print("\nExiting pipeline.")
            break
        except Exception as err:
            print(f"\n[Error]: {err}", file=sys.stderr)

if __name__ == "__main__":
    interactive_cli()
