"""
Configuration & Backend Environment Loader.
Auto-detects Groq vs OpenAI and securely loads API keys and settings from .env.
"""
import os
import sys
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError, APIError

# Load variables from .env file into environment
load_dotenv(override=True)

def get_credentials() -> Tuple[str, str, str]:
    """
    Retrieve API key, base_url, and provider ('groq' or 'openai').
    Auto-detects based on key prefix (gsk_ vs sk-).
    """
    # Accept either GROQ_API_KEY or OPENAI_API_KEY
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key or "your-" in api_key:
        print("\n[ERROR] No valid API key found in .env!", file=sys.stderr)
        print("Please configure GROQ_API_KEY (starts with 'gsk_') or OPENAI_API_KEY in .env\n", file=sys.stderr)
        sys.exit(1)
    
    api_key = api_key.strip()

    if api_key.startswith("gsk_"):
        provider = "groq"
        base_url = "https://api.groq.com/openai/v1"
        default_model = os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-120b"
    else:
        provider = "openai"
        base_url = "https://api.openai.com/v1"
        default_model = os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    return api_key, base_url, default_model, provider

def get_openai_client() -> OpenAI:
    """Create and return an authenticated client for Groq or OpenAI."""
    api_key, base_url, _, _ = get_credentials()
    return OpenAI(api_key=api_key, base_url=base_url)

def get_default_model() -> str:
    _, _, model, _ = get_credentials()
    return model

def get_provider() -> str:
    _, _, _, provider = get_credentials()
    return provider

def get_embedding_generator():
    """Return local fastembed embedding generator for zero-cost, high-speed embeddings."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def handle_openai_error(err: Exception):
    """Provide clear, actionable diagnostics for API errors."""
    provider = get_provider().upper()
    if isinstance(err, RateLimitError):
        print(f"\n[{provider} RateLimit / Quota Notice]:", file=sys.stderr)
        print(f"Your API key was authenticated, but {provider} returned a rate/credit limit error.", file=sys.stderr)
    elif isinstance(err, AuthenticationError):
        print(f"\n[{provider} Auth Error]: Invalid API Key. Please verify your key in .env\n", file=sys.stderr)
    elif isinstance(err, APIError):
        print(f"\n[{provider} API Error]: {err.message}\n", file=sys.stderr)
    else:
        print(f"\n[Unexpected Error]: {err}\n", file=sys.stderr)

DEFAULT_CHAT_MODEL = get_default_model()
