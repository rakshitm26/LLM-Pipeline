"""
Tutorial 5: Complete RAG Pipeline with LlamaIndex (Groq + FastEmbed)
--------------------------------------------------------------------
Concepts covered:
1. Connecting LlamaIndex to Groq ultra-fast LLM
2. Generating local dense embeddings with FastEmbed (no API costs)
3. Reading documents from `data/` and querying private knowledge
"""

import os
import sys
from pathlib import Path
from typing import List, Any
from fastembed import TextEmbedding

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from config import get_credentials, handle_openai_error

class FastEmbedAdapter(BaseEmbedding):
    """Local, zero-cost fast embedding adapter for LlamaIndex."""
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

def main():
    print("=" * 60)
    print("Tutorial 5: RAG Pipeline with LlamaIndex & Groq")
    print("=" * 60)

    try:
        api_key, base_url, model_name, provider = get_credentials()

        # 1. Setup LLM based on provider
        if provider == "groq":
            from llama_index.llms.groq import Groq
            Settings.llm = Groq(model=model_name, api_key=api_key)
        else:
            from llama_index.llms.openai import OpenAI as LlamaOpenAI
            Settings.llm = LlamaOpenAI(model=model_name, api_key=api_key)

        # 2. Setup local zero-cost embeddings
        Settings.embed_model = FastEmbedAdapter()

        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        print(f"Loading knowledge documents from: {os.path.abspath(data_dir)}")

        # 3. Load private documents from data folder
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"Loaded {len(documents)} document chunk(s).")

        # 4. Build Vector Store Index
        print("Indexing documents into local vector store...")
        index = VectorStoreIndex.from_documents(documents)

        # 5. Create Query Engine
        query_engine = index.as_query_engine(similarity_top_k=2)

        query = "What is LlamaIndex and why is it useful?"
        print(f"\nUser Query: {query}")
        print(f"Querying index with {provider.upper()} ({model_name})...\n")

        response = query_engine.query(query)

        print("--- Answer from RAG Pipeline ---")
        print(response.response)

        print("\n--- Source Documents Retrieved ---")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"[Source {i} | Score: {node.score:.4f}] {node.node.get_text().strip()}")
    except Exception as err:
        handle_openai_error(err)

if __name__ == "__main__":
    main()
