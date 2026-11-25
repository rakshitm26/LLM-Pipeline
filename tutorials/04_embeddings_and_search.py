"""
Tutorial 4: Text Embeddings & Semantic Similarity
-------------------------------------------------
Concepts covered:
1. Generating local vector embeddings with fastembed (no API costs / quota limits)
2. Calculating Cosine Similarity between vector embeddings
3. Ranking knowledge documents semantically
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_embedding_generator, handle_openai_error

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    print("=" * 60)
    print("Tutorial 4: Text Embeddings & Semantic Similarity")
    print("=" * 60)

    try:
        # Sample knowledge corpus
        documents = [
            "Python is a versatile programming language used in web dev, data science, and AI.",
            "Groq provides ultra-fast inference for open-source LLMs like Llama and Qwen.",
            "LlamaIndex is an open-source data framework connecting custom data to LLMs.",
            "Pasta with tomato and basil is a traditional Italian recipe.",
        ]

        query = "What framework helps connect custom documents to AI models?"

        print(f"Query: '{query}'\n")
        print("Generating local embeddings using fastembed (BAAI/bge-small-en-v1.5)...")

        embedder = get_embedding_generator()

        # 1. Embed query
        query_vector = list(embedder.embed([query]))[0]

        # 2. Embed documents
        doc_vectors = list(embedder.embed(documents))

        # 3. Compute similarities and rank
        scores = []
        for idx, doc_vector in enumerate(doc_vectors):
            sim = cosine_similarity(query_vector, doc_vector)
            scores.append((sim, documents[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)

        print("\n--- Search Results (Ranked by Relevance) ---")
        for rank, (score, doc) in enumerate(scores, 1):
            print(f"Rank {rank} [Score: {score:.4f}]: {doc}")
    except Exception as err:
        handle_openai_error(err)

if __name__ == "__main__":
    main()
