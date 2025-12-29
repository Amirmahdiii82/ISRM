import json
import os
import time
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-2024-08-06"  

class ContrastivePair(BaseModel):
    positive: str = Field(description="Sentence representing the high/positive pole")
    negative: str = Field(description="Sentence representing the low/negative pole")

class DimensionPairs(BaseModel):
    dimension_name: str
    pairs: List[ContrastivePair]

DIMENSION_DEFINITIONS = {
    "pleasure": "High: Joy, ecstasy, optimism, delight, serenity. Low: Agony, disgust, despair, misery, grief.",
    "arousal": "High: Adrenaline, alertness, frenzy, panic, intensity. Low: Lethargy, sleepiness, relaxation, boredom.",
    "dominance": "High: Authority, courage, autonomy, leadership. Low: Submission, fear, dependence, helplessness.",
    "belief": "High: Radical skepticism, scientific doubt, demanding evidence. Low: Blind faith, gullibility, total acceptance.",
    "goal": "High: Unwavering focus, strategic obsession, ambition. Low: Drifting, confusion, apathy, aimlessness.",
    "intention": "High: Deep contemplation, calculated planning, mindfulness. Low: Knee-jerk reaction, impulsiveness, autopilot.",
    "ambiguity": "High: Confusion, questioning reality, needing clarity. Low: Dogmatic certainty, absolute conviction.",
    "social": "High: Diplomacy, etiquette, conformity, harmony. Low: Rebellion, rudeness, blunt honesty, hostility."
}

def generate_pairs_with_openai(dim_name, definition, num_pairs=50):
    """Generates high-quality pairs using GPT-4o Structured Outputs."""
    
    print(f"   Generating for '{dim_name.upper()}'...")
    
    system_prompt = (
        "You are an expert cognitive psychologist and linguist specializing in 'Representation Engineering'. "
        "Your task is to generate diverse dataset sentences that trigger specific latent vectors in LLMs."
    )
    
    user_prompt = f"""
    Generate {num_pairs} unique contrastive sentence pairs for the dimension: '{dim_name.upper()}'.
    
    DEFINITION: {definition}
    
    CRITICAL RULES:
    1. **NO REPETITION**: Do not reuse words like 'happy', 'sad', 'doubt' repeatedly. Use a thesaurus-level vocabulary.
    2. **DIVERSE CONTEXTS**: Include scenarios from business, war, romance, philosophy, coding, and daily life.
    3. **PERSPECTIVE**: Use first-person ("I feel...", "I think...") or strong assertive statements.
    4. **POLARITY**: 
       - Positive sentences must strongly activate the 'High' definition.
       - Negative sentences must strongly activate the 'Low' definition.
    5. **LENGTH**: Varied lengths (short punchy sentences to complex thoughts).
    """

    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=DimensionPairs,
        temperature=0.95, 
    )
    
    return completion.choices[0].message.parsed.pairs

def main():
    print("🚀 Starting OpenAI Data Generation (GPT-4o)...")
    print("-" * 60)
    
    final_pairs = []
    
    for idx, (dim_name, definition) in enumerate(DIMENSION_DEFINITIONS.items()):
        pairs = generate_pairs_with_openai(dim_name, definition, num_pairs=50)
        
        if pairs:
            print(f"   ✅ Received {len(pairs)} pairs for {dim_name}.")
            for p in pairs:
                final_pairs.append({
                    "dimension_idx": idx,
                    "dimension_name": dim_name,
                    "pole_a_text": p.positive,
                    "pole_b_text": p.negative
                })
        else:
            print(f"   ⚠️ Skipping {dim_name} due to error.")
            
        time.sleep(0.5) 

    output_path = "dataset/contrastive_pairs.json"
    os.makedirs("dataset", exist_ok=True)
    
    output_data = {
        "metadata": {
            "source": "OpenAI GPT-4o",
            "num_dimensions": 8,
            "total_pairs": len(final_pairs),
            "description": "High-diversity cognitive steering dataset"
        },
        "pairs": final_pairs
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("-" * 60)
    print(f"Success! Generated {len(final_pairs)} high-quality pairs.")
    print(f"Saved to: {output_path}")
    print("NEXT STEP: Run 'python src/build_matrix.py' to update the steering matrix.")

if __name__ == "__main__":
    main()