"""
Tutorial 6: Conversational Agent with Memory Pipeline
-----------------------------------------------------
Concepts covered:
1. Implementing multi-turn conversation memory
2. Maintaining conversation context across user prompts
3. Truncating message history to control context window and token spend
"""

import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_openai_client, DEFAULT_CHAT_MODEL, handle_openai_error

class ConversationPipeline:
    """Manages chat conversation state and LLM interactions."""
    def __init__(self, system_prompt: str = "You are a helpful Python mentor.", max_history: int = 10):
        self.client = get_openai_client()
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._prune_history()

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._prune_history()

    def _prune_history(self):
        # Keep system prompt at [0] + last max_history interactions
        if len(self.history) > (self.max_history + 1):
            self.history = [self.history[0]] + self.history[-self.max_history:]

    def ask(self, user_input: str) -> str:
        self.add_user_message(user_input)

        response = self.client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            messages=self.history,
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        self.add_assistant_message(reply)
        return reply

def main():
    print("=" * 60)
    print("Tutorial 6: Multi-Turn Conversational Memory Pipeline")
    print("=" * 60)

    try:
        bot = ConversationPipeline(system_prompt="You are a friendly, witty Python tutor.")

        turns = [
            "Hi, my favorite Python library is pandas.",
            "What did I just say my favorite Python library was?",
            "Can you show me a one-liner to create a DataFrame with columns 'a' and 'b'?",
        ]

        for turn in turns:
            print(f"\nUser: {turn}")
            reply = bot.ask(turn)
            print(f"Assistant: {reply}")
    except Exception as err:
        handle_openai_error(err)

if __name__ == "__main__":
    main()
