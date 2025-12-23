import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

filename = 'dataset/isrm_dataset_final.json' 

with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

if data:
    vectors = []
    labels = []

    for item in data:
        vectors.append(item['state_vector'])
        labels.append(item.get('scenario_category', 'Unknown'))

    X = np.array(vectors)
    
    plt.figure(figsize=(20, 9))
    unique_labels = list(set(labels))
    palette = sns.color_palette("bright", len(unique_labels))

    perp = min(40, len(data) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.7,
        s=60
    )
    plt.title(f"t-SNE (Local Clusters) - Perplexity: {perp}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/amir/Desktop/ISRM/plots/dataset_plots/isrm_visualization_tsne.png')
    print("Plot saved as 'plots/dataset_plots/isrm_visualization_tsne.png'")
    plt.show()