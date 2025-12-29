import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# تنظیمات ظاهری
sns.set(style="whitegrid")

def visualize_contrastive():
    print("Loading Contrastive Pairs...")
    try:
        with open('dataset/contrastive_pairs.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: 'dataset/contrastive_pairs.json' not found.")
        return
    
    pairs = data['pairs']
    
    # آماده‌سازی متن برای آنالیز
    # ما متن مثبت و منفی را ترکیب می‌کنیم تا "مفهوم کلی" را بگیریم
    corpus = [p['pole_a_text'] + " " + p['pole_b_text'] for p in pairs]
    labels = [p['dimension_name'] for p in pairs]
    
    # 1. تبدیل متن به عدد (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(corpus)
    
    # 2. کاهش ابعاد به 2D (t-SNE)
    print("Running t-SNE clustering...")
    # perplexity را بر اساس تعداد نمونه‌ها تنظیم می‌کنیم تا خطا ندهد
    perp = min(30, len(pairs) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X.toarray())
    
    # 3. رسم نمودار کلاسترینگ
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1], 
        hue=labels, style=labels, 
        palette='bright', s=120, alpha=0.8
    )
    plt.title("Semantic Separation of Psychological Dimensions (TF-IDF + t-SNE)", fontsize=16)
    plt.xlabel("Semantic Component 1")
    plt.ylabel("Semantic Component 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/dataset_plots/contrastive_clusters.png')
    print("✅ Saved: contrastive_clusters.png")

    # --- PLOT 2: Balance Check (FIXED) ---
    plt.figure(figsize=(10, 6))
    
    # تغییر اصلی اینجاست: اضافه کردن hue و legend=False
    sns.countplot(
        y=labels, 
        hue=labels,  # مقداردهی صریح hue
        palette='viridis', 
        order=pd.Series(labels).value_counts().index,
        legend=False # حذف لجند تکراری
    )
    
    plt.title("Dataset Balance: Samples per Dimension")
    plt.xlabel("Count")
    plt.ylabel("Dimension")
    plt.tight_layout()
    plt.savefig('plots/dataset_plots/contrastive_balance.png')
    print("✅ Saved: contrastive_balance.png")

if __name__ == "__main__":
    visualize_contrastive()