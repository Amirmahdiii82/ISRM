![ISRM Logo](assets/logo.jpg)


# 🧠 ISRM: Internal State Reasoning Module

### Steerable Open-Endedness in LLMs via Variational Latent State Modeling

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/)

**ISRM** is a novel "Sidecar Architecture" that decouples an agent's **internal psychological state** from its **linguistic generation**. By leveraging **Representation Engineering (RepE)**, ISRM injects continuous, psychometrically valid latent vectors (PAD & BDI models) directly into the hidden layers of a frozen Large Language Model. This enables precise neural-level steerability and behavioral diversity without the need for expensive fine-tuning of the base LLM.

-----

## 🚀 Key Features

  * **🧠 Decoupled Brain & Body:** Uses a trainable **VAE Encoder** (DistilBERT-based) to model "feelings" and a frozen **LLM Body** (Qwen3-4B-Thinking) to express them.
  * **⚡ RepE Neural Steering:** Direct injection of steering vectors into LLM hidden layers (layer 14 by default) for precise behavioral control.
  * **🎛️ Geometric Steering:** Control agent behavior via an 8-dimensional continuous latent space (Pleasure, Arousal, Dominance, Belief, Goal, Intention, Ambiguity, Social).

-----

## 🏗️ Architecture

The system combines **Representation Engineering (RepE)** with a VAE-based state encoder:

1. **ISRM Encoder (The Brain):** A fine-tuned DistilBERT VAE that maps dialogue context to an 8D psychological state vector `z ∈ [0,1]^8`
2. **Steering Matrix (The Bridge):** An 8×hidden_dim matrix extracted from contrastive pairs using Mean Difference method
3. **Neural Injection (The Control):** The product `z × Matrix` is injected into layer 14 of the frozen LLM during forward pass
4. **LLM Generator (The Body):** Qwen3-4B-Thinking generates responses influenced by the injected state

This approach enables **neural-level steering** without modifying model weights or requiring system prompts.

-----

## 🛠️ Installation

### Prerequisites

  * Python 3.10+
  * CUDA 12.x (Recommended for inference)
  * PyTorch 2.0+

### Option A: Quick Start with Pre-trained Models (Recommended)

**1. Clone the Repository**

```bash
git clone https://github.com/your-username/ISRM.git
cd ISRM
```

**2. Install Dependencies**

```bash
pip install torch transformers sentence-transformers huggingface_hub rich
```

**3. Download Pre-trained Models from Hugging Face** 🤗

```bash
# Install huggingface-cli if needed
pip install -U huggingface_hub

# Download ISRM encoder and steering matrix
python -c "
from huggingface_hub import hf_hub_download
import os

os.makedirs('model/isrm', exist_ok=True)
os.makedirs('vectors', exist_ok=True)

# Download encoder
hf_hub_download(
    repo_id='YOUR_USERNAME/isrm',
    filename='pad_encoder.pth',
    local_dir='model/isrm'
)

# Download steering matrix
hf_hub_download(
    repo_id='YOUR_USERNAME/isrm',
    filename='steering_matrix.pt',
    local_dir='vectors'
)

print('✓ Models downloaded successfully!')
"
```

**4. Download Base LLM (One-time)**

```bash
# Download Qwen3-4B-Thinking to local cache
python scripts/download_llm.py
```

**5. Run the Agent**

```bash
# Interactive chat with ISRM agent
python src/chat.py --persona skeptical
```

### Option B: Train from Scratch

If you want to reproduce the training:

**1. Generate Training Data**

```bash
# Generate contrastive pairs for RepE (requires API key)
python src/data_gen.py

# Generate PAD training dataset
python scripts/data_generator.py
```

**2. Build Steering Matrix**

```bash
# Extract steering vectors from LLM using RepE Mean Difference
python src/build_matrix.py
```

This creates `vectors/steering_matrix.pt` (8×hidden_dim matrix).

**3. Train ISRM Encoder**

```bash
# Train the VAE encoder on PAD dataset
python src/train.py
```

This creates `model/isrm/pad_encoder.pth`.

**4. Run Scientific Validation** (Optional)

```bash
# Validate with ActAdd + PSYA metrics (n=5 trials, ~20-30 min)
python src/validation_scientific.py
```

-----

## 📂 Project Structure

### Core Source Files (`src/`)

| File | Description |
| :--- | :--- |
| `model.py` | **The Brain.** Contains the `ISRM_Architected` class (VAE Encoder). |
| `alignment.py` | **The Bridge.** Implements `NeuralAgent` with RepE injection hooks and LLM loading. |
| `build_matrix.py` | **RepE Extractor.** Builds steering matrix from contrastive pairs using Mean Difference. |
| `data_gen.py` | **Contrastive Pairs.** Generates cognitive contrastive pairs for RepE training. |
| `chat.py` | **Interaction.** Rich terminal chat interface with state visualization. |
| `train.py` | **Training.** Fine-tunes the ISRM VAE module on dialogue data. |

### Scripts (`scripts/`)

| File | Description |
| :--- | :--- |
| `download_llm.py` | Downloads and caches LLM models to `model/llm/`. |
| `data_generator.py` | Generates synthetic training dataset using Gemini API. |
| `dataset_2d_plot.py` | Visualizes dataset using t-SNE dimensionality reduction. |
| `dataset_radar_plot.py` | Creates radar charts of scenario state fingerprints. |


### Data & Models

| Path | Description | Size | Available on HF |
| :--- | :--- | :--- | :--- |
| `model/isrm/pad_encoder.pth` | Trained ISRM VAE weights | 265MB | ✅ |
| `vectors/steering_matrix.pt` | RepE steering matrix (8×hidden_dim) | 42KB | ✅ |
| `dataset/pad_training_data.json` | PAD training dataset | 1.5MB | ❌ |
| `dataset/contrastive_pairs.json` | Contrastive pairs for RepE | 97KB | ✅ |

**Download from HuggingFace:** `YOUR_USERNAME/isrm`


-----

## 🧠 Methodology

### The 8-Dimensional Latent Space

Our encoder maps dialogue history to a vector `z ∈ [0,1]^8`, derived from:

**PAD Model (Affective):**
1. **Pleasure (P):** Happiness vs. Displeasure [0=Disgusted, 1=Delighted]
2. **Arousal (A):** Energy level [0=Calm/Sleepy, 1=Energetic/Alert]
3. **Dominance (D):** Control perception [0=Submissive, 1=Commanding]

**Cognitive State (BDI-inspired):**
4. **Belief Confidence:** Trust in information [0=Trusting, 1=Skeptical]
5. **Goal Commitment:** Focus on objectives [0=Aimless, 1=Laser-focused]
6. **Intention Stability:** Analytical depth [0=Surface-level, 1=Deep analysis]
7. **Ambiguity Tolerance:** Certainty level [0=Uncertain, 1=Certain]
8. **Social Adherence:** Politeness [0=Blunt, 1=Respectful]

### RepE Steering Mechanism

The steering process follows these steps:

1. **State Encoding:** ISRM VAE encodes dialogue context → `z ∈ [0,1]^8`
2. **Normalization:** Transform to bipolar range: `z_norm = (z - 0.5) × 2 ∈ [-1,1]^8`
3. **Matrix Multiplication:** Compute injection vector: `v = z_norm @ steering_matrix` (shape: hidden_dim)
4. **Layer Injection:** Register forward hook on layer 14 to add `v` to all token activations
5. **Generation:** LLM generates response with modified activations

The steering matrix is extracted once using contrastive pairs and the Mean Difference method from RepE.

### Workflow Diagram

```
User Input → ISRM VAE → z-vector [8D] → Steering Matrix → Injection Vector [hidden_dim]
                                                                      ↓
                                                            Forward Hook @ Layer 14
                                                                      ↓
                                                      LLM Hidden States += Injection
                                                                      ↓
                                                              Generated Response
```

-----

## 🔬 Scientific Validation

ISRM has been validated using rigorous vector-based metrics (not naive keyword counting):

### ActAdd Validation (Sentiment Probability Shift)

Measures `ΔS = P(positive|STEERED) - P(positive|BASE)` using transformer logits:

| Condition | P(pos\|BASE) | P(pos\|STEERED) | ΔS | Cohen's d | p-value |
|-----------|-------------|----------------|-----|-----------|---------|
| **High Pleasure** | 0.530 ± 0.042 | **0.785 ± 0.048** | **+0.255** | 4.58 | <0.001*** |

### PSYA Validation (Semantic Alignment)

Measures cosine similarity between response and persona anchor using SentenceTransformer:

| Persona | Sim(BASE↔Anchor) | Sim(STEERED↔Anchor) | Δ Similarity | Cohen's d | p-value |
|---------|-----------------|---------------------|--------------|-----------|---------|
| **Skeptical** | 0.452 ± 0.038 | **0.687 ± 0.042** | **+0.235** | 4.82 | <0.001*** |

### Controllability (Monotonicity)

Spearman correlation: **ρ = 0.975**, p = 0.001 ✓

Steering magnitude increases monotonically with PAD values, confirming predictable control.

**See [`VALIDATION_GUIDE.md`](VALIDATION_GUIDE.md) for complete methodology.**

-----

## 📜 Citation

If you use this code or architecture, please cite:

```bibtex

```
