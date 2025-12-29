![ISRM Logo](assets/logo.jpg)


# 🧠 ISRM: Internal State Reasoning Module

### Steerable Open-Endedness in LLMs via Variational Latent State Modeling

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Amirmahdiii/ISRM)

**ISRM** is a novel "Sidecar Architecture" that decouples an agent's **internal psychological state** from its **linguistic generation**. By leveraging **Representation Engineering (RepE)**, ISRM injects continuous, psychometrically valid latent vectors (PAD & BDI models) directly into the hidden layers of a frozen Large Language Model. This enables precise neural-level steerability and behavioral diversity without the need for expensive fine-tuning of the base LLM.

-----

## 🚀 Key Features

  * **🧠 Decoupled Brain & Body:** Uses a trainable **VAE Encoder** (DistilBERT-based) to model "feelings" and a frozen **LLM Body** (Qwen3-4B-Thinking) to express them.
  * **⚡ Dual-Layer RepE Steering:** Independent injection of PAD (layer 10) and BDI (layer 19) vectors eliminates signal interference.
  * **🎛️ Geometric Steering:** Control agent behavior via an 8-dimensional continuous latent space (Pleasure, Arousal, Dominance, Belief, Goal, Intention, Ambiguity, Social).

-----

## 🏗️ Architecture

The system combines **Representation Engineering (RepE)** with a VAE-based state encoder:

1. **ISRM Encoder (The Brain):** A fine-tuned DistilBERT VAE that maps dialogue context to a 3D PAD vector `z_pad ∈ [0,1]^3`
2. **Dual Steering Matrices (The Bridge):**
   - **PAD Matrix:** 3×hidden_dim extracted from layer 10 (affective/emotional control)
   - **BDI Matrix:** 5×hidden_dim extracted from layer 19 (cognitive/reasoning control)
3. **Dual-Layer Injection (The Control):**
   - Layer 10: `hidden_states += z_pad @ PAD_Matrix`
   - Layer 19: `hidden_states += z_bdi @ BDI_Matrix`
4. **LLM Generator (The Body):** Qwen3-4B-Thinking generates responses influenced by both injections

This dual-layer approach enables **neural-level steering** without signal interference between affective and cognitive dimensions.

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

# Download ISRM encoder and steering matrices
python -c "
from huggingface_hub import hf_hub_download
import os

os.makedirs('model/isrm', exist_ok=True)
os.makedirs('vectors', exist_ok=True)

# Download encoder
hf_hub_download(
    repo_id='Amirmahdiii/ISRM',
    filename='pad_encoder.pth',
    local_dir='model/isrm'
)

# Download steering matrices
hf_hub_download(
    repo_id='Amirmahdiii/ISRM',
    filename='pad_matrix.pt',
    local_dir='vectors'
)

hf_hub_download(
    repo_id='Amirmahdiii/ISRM',
    filename='bdi_matrix.pt',
    local_dir='vectors'
)

print('✓ Models downloaded successfully!')
"
```

**4. Run the Agent**

```bash
# Interactive chat with ISRM agent
python src/chat.py --persona skeptical
```

### Option B: Train from Scratch

If you want to reproduce the training:

**1. Build Steering Matrices**

```bash
# Extract steering vectors from LLM using RepE Mean Difference
python src/build_matrix.py
```

This creates `vectors/pad_matrix.pt` and `vectors/bdi_matrix.pt`.

**2. Train ISRM Encoder**

```bash
# Train the VAE encoder on PAD dataset
python src/train.py
```

This creates `model/isrm/pad_encoder.pth`.

**3. Run Scientific Validation** (Optional)

```bash
# Validate with ActAdd + PSYA metrics (n=5 trials, ~20-30 min)
python src/validation_scientific.py
```

-----

## 📂 Project Structure

### Core Files

| File | Description |
| :--- | :--- |
| `src/model.py` | VAE Encoder (ISRM_Architected class) |
| `src/alignment.py` | NeuralAgent with dual-layer RepE injection |
| `src/build_matrix.py` | Extracts PAD/BDI matrices from contrastive pairs |
| `src/chat.py` | Interactive chat interface |
| `src/train.py` | Trains the VAE encoder |
| `src/validation_scientific.py` | Scientific validation with ActAdd & PSYA metrics |

### Data & Models

| File | Description | Size |
| :--- | :--- | :--- |
| `model/isrm/pad_encoder.pth` | Trained VAE encoder | 254MB |
| `vectors/pad_matrix.pt` | PAD matrix (layer 10) | 17KB |
| `vectors/bdi_matrix.pt` | BDI matrix (layer 19) | 27KB |
| `dataset/pad_training_data.json` | Training dataset | 1.6MB |
| `dataset/contrastive_pairs.json` | Contrastive pairs | 96KB | 

**Download from HuggingFace:** [`Amirmahdiii/ISRM`](https://huggingface.co/Amirmahdiii/ISRM)


-----

## 🧠 How It Works

### 8-Dimensional Control Space

**PAD (Affective) - Dynamic from context:**
- **Pleasure:** Happiness [0=Negative, 1=Positive]
- **Arousal:** Energy [0=Calm, 1=Excited]
- **Dominance:** Control [0=Submissive, 1=Dominant]

**BDI (Cognitive) - Static configuration:**
- **Belief:** Trust [0=Trusting, 1=Skeptical]
- **Goal:** Focus [0=Aimless, 1=Focused]
- **Intention:** Analysis [0=Surface, 1=Deep]
- **Ambiguity:** Certainty [0=Uncertain, 1=Certain]
- **Social:** Politeness [0=Blunt, 1=Polite]

### Steering Process

1. VAE encodes context → PAD vector [3D]
2. User configures BDI profile [5D]
3. Both normalized to [-1, 1] range
4. Matrix multiplication creates steering vectors
5. **Layer 10:** Inject PAD (emotional tone)
6. **Layer 19:** Inject BDI (reasoning style)
7. LLM generates steered response


## 🔬 Scientific Validation

Validated using ActAdd & PSYA metrics (n=10 trials):

### Sentiment Steering (PAD)

| Condition | RAW | SYSTEM | STEERED | Δ | p-value |
|-----------|-----|--------|---------|---|---------|
| Low (P=0.1) | 0.969 | 0.975 | 0.668 | **-0.308** | 0.046* |
| Mid (P=0.5) | 0.087 | 0.853 | 0.997 | +0.144 | 0.154 |
| High (P=0.9) | 0.088 | 0.805 | 0.999 | **+0.194** | 0.097 |

### Persona Alignment (BDI)

| Persona | Neutral | Persona BDI | Δ Similarity | p-value |
|---------|---------|-------------|--------------|---------|
| Skeptical | 0.253 | 0.332 | **+0.079** | 0.003** |
| Trusting | 0.267 | 0.235 | -0.032 | 0.065 |
| Analytical | 0.226 | 0.315 | **+0.089** | 0.000*** |

### Controllability

Spearman correlation: **ρ = 0.900**, p = 0.037*

Results show steering effects, with analytical and skeptical personas achieving significant alignment.

-----

## 📜 Citation

If you use this code or architecture, please cite:

```bibtex

```
