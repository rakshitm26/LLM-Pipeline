"""
Tutorial 1: Basic LLM Completion & Prompting
--------------------------------------------
Concepts covered:
1. Loading backend credentials from .env via config.py
2. Initializing the OpenAI client
3. Setting System, User, and Assistant roles
4. Adjusting temperature for deterministic vs creative output
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_openai_client, DEFAULT_CHAT_MODEL, handle_openai_error

def main():
    print("=" * 60)
    print("Tutorial 1: Basic Completion with OpenAI Backend")
    print("=" * 60)

    try:
        # 1. Initialize client using the backend .env key
        client = get_openai_client()

        # 2. Build messages payload
        system_prompt = "You are a concise, helpful Python coding assistant."
        user_prompt = "Explain what a Python generator is in 2 sentences with 1 tiny code example."

        print(f"\nModel: {DEFAULT_CHAT_MODEL}")
        print(f"System: {system_prompt}")
        print(f"User: {user_prompt}\n")

        # 3. Request completion from LLM
        response = client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            temperature=0.3,  # Lower = more deterministic, Higher = more creative
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # 4. Extract reply content and usage stats
        reply = response.choices[0].message.content
        usage = response.usage

        print("--- Response ---")
        print(reply)
        print("\n--- Token Usage ---")
        print(f"Prompt Tokens: {usage.prompt_tokens}, Completion Tokens: {usage.completion_tokens}, Total: {usage.total_tokens}")
    except Exception as err:
        handle_openai_error(err)

if __name__ == "__main__":
    main()
