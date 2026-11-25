"""
Tutorial 3: Structured Outputs with Pydantic
--------------------------------------------
Concepts covered:
1. Enforcing strict schema outputs from the LLM
2. Parsing unstructured text into type-safe Python models
3. Preventing schema hallucination for backend APIs and pipelines
"""

import json
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_openai_client, DEFAULT_CHAT_MODEL, handle_openai_error

# Define the schema using Pydantic
class TutorialStep(BaseModel):
    step_number: int = Field(description="Sequential order of the step")
    title: str = Field(description="Short title for the step")
    code_snippet: str = Field(description="Self-contained Python code snippet")
    explanation: str = Field(description="One-sentence explanation of what the step achieves")

class TutorialPlan(BaseModel):
    topic: str = Field(description="Name of the technical topic")
    difficulty: str = Field(description="Beginner, Intermediate, or Advanced")
    prerequisites: List[str] = Field(description="Key prerequisites required before starting")
    steps: List[TutorialStep] = Field(description="Detailed sequential steps")

def main():
    print("=" * 60)
    print("Tutorial 3: Structured Output Extraction (Pydantic)")
    print("=" * 60)

    try:
        client = get_openai_client()
        topic = "Connecting to SQLite and reading data with Python"

        print(f"\nRequesting structured tutorial plan for: '{topic}'...")

        # Using client.beta.chat.completions.parse for schema enforcement
        completion = client.beta.chat.completions.parse(
            model=DEFAULT_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a curriculum developer creating structured Python tutorials.",
                },
                {
                    "role": "user",
                    "content": f"Create a concise 2-step tutorial for: {topic}",
                },
            ],
            response_format=TutorialPlan,
        )

        tutorial: TutorialPlan = completion.choices[0].message.parsed

        print("\n--- Parsed Pydantic Model Output ---")
        print(f"Topic:        {tutorial.topic}")
        print(f"Difficulty:   {tutorial.difficulty}")
        print(f"Prerequisites: {', '.join(tutorial.prerequisites)}")
        print("\nSteps:")
        for step in tutorial.steps:
            print(f"\n  [Step {step.step_number}] {step.title}")
            print(f"  Explanation: {step.explanation}")
            print("  Code:")
            for line in step.code_snippet.strip().splitlines():
                print(f"    {line}")

        print("\n--- Serialized JSON for Backend API ---")
        print(json.dumps(tutorial.model_dump(), indent=2))
    except Exception as err:
        handle_openai_error(err)

if __name__ == "__main__":
    main()
