"""
Tutorial 2: Real-time Streaming Responses
-----------------------------------------
Concepts covered:
1. Using stream=True for instant, low-latency UI feedback
2. Iterating through response chunks (Server-Sent Events style)
3. Handling token generation in real-time
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_openai_client, DEFAULT_CHAT_MODEL, handle_openai_error

def main():
    print("=" * 60)
    print("Tutorial 2: Real-Time Streaming Responses")
    print("=" * 60)

    try:
        client = get_openai_client()

        prompt = "Write a short haiku about clean code and explain the meaning."
        print(f"\nUser Prompt: {prompt}\n")
        print("Streaming output:\n")

        # Request with streaming enabled
        stream = client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a creative technical poet."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        # Print tokens as they arrive
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()

        print("\n\n[Stream Completed]")
    except Exception as err:
        handle_openai_error(err)

if __name__ == "__main__":
    main()
