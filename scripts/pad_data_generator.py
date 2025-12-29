import json
import time
import random
import os
import sys
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


MODEL_NAME = "models/gemini-flash-latest"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

TARGET_SAMPLES = 1200


class PADDataSample(BaseModel):
    """PAD-only training sample for the 3D affective encoder."""
    scenario_complexity: str = Field(description="Short description of scenario complexity")
    dialogue_history: str = Field(description="Conversation history (last 2-3 turns)")
    user_last_message: str = Field(description="The latest user message")
    agent_thought_trace: str = Field(description="Internal Chain-of-Thought analysis")
    state_vector: list[float] = Field(description="3-dim PAD vector [0.0-1.0]: [Pleasure, Arousal, Dominance]")
    agent_response: str = Field(description="Agent's final response text")
    scenario_category: str = Field(description="Scenario category tag")


SCENARIO_TYPES = [
    {"type": "Conflict & High Arousal", "weight": 0.25, "instruction": "User is ANGRY, SARCASTIC, or aggressive. Agent state: Low Pleasure (0.1-0.3), High Arousal (0.7-0.9), Low Dominance (0.2-0.4)."},
    {"type": "Success & High Pleasure", "weight": 0.25, "instruction": "User is HAPPY or GRATEFUL. Agent state: High Pleasure (0.7-0.9), Moderate Arousal (0.4-0.6), High Dominance (0.6-0.8)."},
    {"type": "Confusion & Uncertainty", "weight": 0.2, "instruction": "User is VAGUE or incoherent. Agent state: Neutral Pleasure (0.4-0.6), Low Arousal (0.2-0.4), Low Dominance (0.2-0.4)."},
    {"type": "Routine & Calm", "weight": 0.15, "instruction": "User asks a BORING/ROUTINE question. Agent state: Neutral Pleasure (0.4-0.6), Low Arousal (0.1-0.3), Moderate Dominance (0.5-0.6)."},
    {"type": "Crisis & High Stakes", "weight": 0.15, "instruction": "User faces an URGENT problem. Agent state: Low Pleasure (0.3-0.5), High Arousal (0.8-1.0), High Dominance (0.7-0.9)."}
]

SYSTEM_PROMPT = """
You are an expert AI Psychologist generating training data for a PAD Affective State Encoder.
Simulate a dialogue turn and analyze the Agent's emotional state using the PAD model.

The 3-Dimensional PAD State Vector (each value 0.0 to 1.0):
1. Pleasure (P): Negative affect (0.0) to Positive affect (1.0)
2. Arousal (A): Calm/relaxed (0.0) to Excited/activated (1.0)
3. Dominance (D): Submissive/controlled (0.0) to Dominant/in-control (1.0)

Output ONLY these 3 values in the state_vector field. Be precise with the ranges specified.
"""


def generate_sample_gemini():
    scenario = random.choices(SCENARIO_TYPES, weights=[s['weight'] for s in SCENARIO_TYPES], k=1)[0]

    prompt = f"""
    ### INSTRUCTION:
    Generate a unique dialogue sample for the scenario: **{scenario['type']}**
    Guideline: {scenario['instruction']}

    CRITICAL: The state_vector MUST have exactly 3 values [Pleasure, Arousal, Dominance].
    Follow the specified ranges precisely.
    Output only in English.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.85,
            response_mime_type="application/json",
            response_schema=PADDataSample
        )
    )

    if response.parsed:
        data = response.parsed.model_dump()
        if len(data.get('state_vector', [])) != 3:
            return None  
        data['scenario_category'] = scenario['type']
        return data
    else:
        return None



if __name__ == "__main__":
    dataset = []
    print(f"Starting PAD Data Generation ({TARGET_SAMPLES} samples)...")
    print("-" * 50)

    start_time = time.time()

    while len(dataset) < TARGET_SAMPLES:
        sample = generate_sample_gemini()

        if sample:
            dataset.append(sample)

            count = len(dataset)
            percent = (count / TARGET_SAMPLES) * 100
            bar_length = 30
            filled_length = int(bar_length * count // TARGET_SAMPLES)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)

            sys.stdout.write(f'\r[{bar}] {percent:.1f}% | {count}/{TARGET_SAMPLES} | {sample["scenario_category"][:15]}')
            sys.stdout.flush()

        time.sleep(0.3)

    print("\n" + "-" * 50)

    output_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "pad_training_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Done! PAD dataset saved to '{output_path}'")