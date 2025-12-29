import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set(style="whitegrid")

def visualize_pad():
    print("Loading PAD Data...")
    with open('dataset/pad_training_data.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df[['Pleasure', 'Arousal', 'Dominance']] = pd.DataFrame(df['state_vector'].tolist(), index=df.index)

    plt.figure(figsize=(10, 9))
    sns.scatterplot(
        data=df, x='Pleasure', y='Arousal', 
        hue='scenario_category', style='scenario_category', 
        palette='deep', s=100, alpha=0.7
    )
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title("Russell's Circumplex Projection (Valence vs Arousal)", fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig('plots/dataset_plots/pad_circumplex.png')
    print("✅ Saved: pad_circumplex.png")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    categories = df['scenario_category'].unique()
    colors = sns.color_palette("husl", len(categories))
    
    for cat, color in zip(categories, colors):
        subset = df[df['scenario_category'] == cat]
        ax.scatter(subset['Pleasure'], subset['Arousal'], subset['Dominance'], 
                   label=cat, s=60, color=color, alpha=0.6)
    
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title('3D Affective Space Analysis', fontsize=16)
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
    plt.savefig('plots/dataset_plots/pad_3d_scatter.png')
    print("✅ Saved: pad_3d_scatter.png")

if __name__ == "__main__":
    visualize_pad()