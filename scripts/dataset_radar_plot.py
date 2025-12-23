import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

filename = 'isrm_dataset_final.json'

with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

vec_labels = ["Pleasure", "Arousal", "Dominance", "Belief Conf.", "Goal Commit.", "Intention Stab.", "Ambiguity Tol.", "Social Adher."]

df[vec_labels] = pd.DataFrame(df['state_vector'].tolist(), index=df.index)

category_means = df.groupby('scenario_category')[vec_labels].mean()

print(category_means)

def plot_radar(category_means):
    categories = list(category_means.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 
    
    scenario_colors = {
        "Conflict & High Arousal": "red",
        "Success & High Pleasure": "green",
        "Confusion & Low Confidence": "blue",
        "Ethical Dilemma": "purple",
        "Routine & Low Arousal": "gray"
    }

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='black', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.0)
    
    for idx, row in category_means.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        
        color = scenario_colors.get(idx, "black") 
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=idx, color=color)
        ax.fill(angles, values, color=color, alpha=0.05) 

    plt.title("ISRM State Fingerprints (Scenario Analysis)", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig('plots/isrm_radar_analysis.png')
    print("\n✅ Radar chart saved as 'plots/isrm_radar_analysis.png'")
    plt.show()

plot_radar(category_means)