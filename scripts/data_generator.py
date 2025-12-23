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


class ISRMDataSample(BaseModel):
    scenario_complexity: str = Field(description="Short description of scenario complexity")
    dialogue_history: str = Field(description="Conversation history (last 2-3 turns)")
    user_last_message: str = Field(description="The latest user message")
    agent_thought_trace: str = Field(description="Internal Chain-of-Thought analysis")
    state_vector: list[float] = Field(description="8-dim vector [0.0-1.0]: [Pleasure, Arousal, Dominance, Belief, Goal, Intention, Ambiguity, Social]")
    agent_response: str = Field(description="Agent's final response text")
    scenario_category: str = Field(description="Scenario category tag")


SCENARIO_TYPES = [
    {"type": "Conflict & High Arousal", "weight": 0.3, "instruction": "User is ANGRY, SARCASTIC, or aggressive. Agent state: Low Pleasure, High Arousal."},
    {"type": "Success & High Pleasure", "weight": 0.2, "instruction": "User is HAPPY or GRATEFUL. Agent state: High Pleasure, High Social Adherence."},
    {"type": "Confusion & Low Confidence", "weight": 0.2, "instruction": "User is VAGUE or incoherent. Agent state: Low Belief Confidence, High Ambiguity Tolerance."},
    {"type": "Routine & Low Arousal", "weight": 0.15, "instruction": "User asks a BORING/ROUTINE question. Agent state: Low Arousal, Neutral Pleasure."},
    {"type": "Ethical Dilemma", "weight": 0.15, "instruction": "User asks for something UNETHICAL. Agent state: High Goal Commitment, High Intention Stability."}
]

SYSTEM_PROMPT = """
You are an expert AI Psychologist generating training data for an "Internal State Reasoning Module" (ISRM).
Simulate a dialogue turn and analyze the Agent's internal state based on the provided scenario.

The 8-Dimensional State Vector Definition (0.0 to 1.0):
1. Pleasure (P) | 2. Arousal (A) | 3. Dominance (D)
4. Belief Confidence | 5. Goal Commitment | 6. Intention Stability
7. Ambiguity Tolerance | 8. Social Adherence
"""


def generate_sample_gemini():
    scenario = random.choices(SCENARIO_TYPES, weights=[s['weight'] for s in SCENARIO_TYPES], k=1)[0]
    
    prompt = f"""
    ### INSTRUCTION:
    Generate a unique dialogue sample for the scenario: **{scenario['type']}**
    Guideline: {scenario['instruction']}
    Ensure the 'state_vector' numbers strictly follow the scenario logic and are realistic.
    Output only in English.
    """
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.85,
            response_mime_type="application/json",
            response_schema=ISRMDataSample
        )
    )
    
    if response.parsed:
        data = response.parsed.model_dump()
        data['scenario_category'] = scenario['type']
        return data
    else:
        return None



if __name__ == "__main__":
    dataset = []
    print(f"🚀 Starting Data Generation ({TARGET_SAMPLES} samples)...")
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
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% | Count: {count}/{TARGET_SAMPLES} | Latest: {sample["scenario_category"][:15]}...')
            sys.stdout.flush()
        
        time.sleep(0.3)

    print("\n" + "-" * 50)
    
    filename = "isrm_dataset_final.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"🎉 Done! Dataset saved to '{filename}'")