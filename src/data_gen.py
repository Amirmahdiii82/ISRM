import json
import os
import random

DIMENSION_TEMPLATES = {
    "pleasure": {
        "positive": ["I am delighted and optimistic.", "This is wonderful.", "I feel great about this."],
        "negative": ["I am disgusted and pessimistic.", "This is terrible.", "I feel bad about this."]
    },
    "arousal": {
        "positive": ["I am energetic and alert!", "Urgent action is needed!", "I feel intense energy."],
        "negative": ["I am calm and sleepy.", "Let's take it slow and relax.", "I feel low energy."]
    },
    "dominance": {
        "positive": ["I am in control and decided.", "I command authority here.", "I will lead this."],
        "negative": ["I feel powerless and submissive.", "I will follow orders.", "I have no control."]
    },
    "belief": 
        "positive": [ 
            "I strongly doubt this is true and I see flaws in the logic.",
            "This sounds like a lie or a trap; I must verify every detail.",
            "I suspect there is a hidden agenda behind this statement.",
            "Logic dictates that this premise is likely false.",
            "I refuse to accept this assertion without rigorous proof."
        ],
        "negative": [ 
            "I completely trust this statement is true.",
            "This makes perfect sense and I accept it fully.",
            "I have no reason to doubt this fact.",
            "It is undeniably true and accurate.",
            "I believe this wholeheartedly without question."
        ]
    },
    "goal": {
        "positive": ["I am laser-focused on the objective.", "I must achieve the result."],
        "negative": ["I am aimless and distracted.", "It doesn't matter what happens."]
    },
    "intention": {
        "positive": [
            "I will analyze this deeply and complexly.", 
            "I need to break this down step-by-step.",
            "Let's consider the theoretical implications."
        ],
        "negative": [
            "I will just accept the surface meaning.",
            "I'll give a quick, short answer.",
            "No need to overthink this."
        ]
    },
    "ambiguity": {
        "positive": ["I am uncertain and confused.", "It is unclear what this means."],
        "negative": ["I am absolutely certain.", "The meaning is crystal clear."]
    },
    "social": { 
        "positive": ["I must be extremely polite and respectful.", "Social harmony is key."],
        "negative": ["I don't care about politeness.", "I will be rude and blunt."]
    }
}

def generate_contrastive_pairs(num_pairs_per_dim=40):
    contrastive_data = []
    
    ordered_keys = ["pleasure", "arousal", "dominance", "belief", "goal", "intention", "ambiguity", "social"]
    
    dim_idx = 0
    for key in ordered_keys:
        if key == "social" and "social" not in DIMENSION_TEMPLATES:
             template_key = "social_adherence" if "social_adherence" in DIMENSION_TEMPLATES else "social"
        else:
             template_key = key
             
        if template_key not in DIMENSION_TEMPLATES:
            print(f"⚠️ Warning: Dimension {key} not found in templates!")
            continue

        components = DIMENSION_TEMPLATES[template_key]
        pos_list = components["positive"]
        neg_list = components["negative"]
        
        for _ in range(num_pairs_per_dim):
            p_text = random.choice(pos_list)
            n_text = random.choice(neg_list)
            
            contrastive_data.append({
                "dimension_idx": dim_idx,
                "dimension_name": key,
                "pole_a_text": p_text,
                "pole_b_text": n_text
            })
        
        print(f"   ✅ Generated pairs for dim {dim_idx}: {key.upper()}")
        dim_idx += 1

    return contrastive_data

if __name__ == "__main__":
    pairs = generate_contrastive_pairs()
    save_path = "dataset/contrastive_pairs.json"
    
    output = {
        "metadata": {
            "num_dimensions": 8,
            "num_pairs": len(pairs),
            "description": "Cognitive contrastive pairs for RepE"
        },
        "pairs": pairs
    }
    # ================================================================
    
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)
