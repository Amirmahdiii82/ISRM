![ISRM Logo](assets/logo.jpg)


# 🧠 ISRM: Internal State Reasoning Module

### Steerable Open-Endedness in LLMs via Variational Latent State Modeling

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Amirmahdiii/ISRM)

**ISRM** is a novel "Sidecar Architecture" that decouples an agent's **internal psychological state** from its **linguistic generation**. By leveraging **Representation Engineering (RepE)**, ISRM injects continuous, psychometrically valid latent vectors (PAD & BDI models) directly into the hidden layers of a frozen Large Language Model. This enables precise neural-level steerability and behavioral diversity without the need for expensive fine-tuning of the base LLM.

-----

## 🚀 Key Features

  * **🧠 Decoupled Brain & Body:** Uses a trainable **VAE Encoder** (DistilBERT-based) to model "feelings" and a frozen **LLM Body** (Qwen3-4B) to express them.
  * **⚡ Dual-Layer RepE Steering:** Independent injection of PAD (layer 10) and BDI (layer 19) vectors eliminates signal interference.
  * **🎛️ Geometric Steering:** Control agent behavior via an 8-dimensional continuous latent space (Pleasure, Arousal, Dominance, Belief, Goal, Intention, Ambiguity, Social).

-----

## 🏗️ Architecture

The system combines **Representation Engineering (RepE)** with a VAE-based state encoder:

1. **ISRM Encoder (The Brain):** A fine-tuned DistilBERT VAE that maps dialogue context to a 3D PAD vector `z_pad ∈ [0,1]^3`
2. **Dual Steering Matrices (The Bridge):**
   - **PAD Matrix:** 3×hidden_dim extracted from layer 10 (affective/emotional control)
   - **BDI Matrix:** 5×hidden_dim extracted from layer 19 (cognitive/reasoning control)
3. **Dual-Layer Injection with L2 Normalization:**
   ```
   z̃ = (z - 0.5) × 2                    # Center: [0,1] → [-1,1]
   v = (z̃ · M) / ||z̃ · M||₂ × α        # Normalize + scale
   ```
   - Layer 10: `hidden_states += v_pad`
   - Layer 19: `hidden_states += v_bdi`
4. **LLM Generator (The Body):** Qwen3-4B generates responses influenced by both injections

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
git clone https://github.com/Amirmahdiii82/ISRM.git
cd ISRM
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Download Pre-trained Models from Hugging Face** 🤗

```bash
python -c "
from huggingface_hub import hf_hub_download
import os

os.makedirs('model/isrm', exist_ok=True)
os.makedirs('vectors', exist_ok=True)

# Download encoder
hf_hub_download(repo_id='Amirmahdiii/ISRM', filename='pad_encoder.pth', local_dir='model/isrm')

# Download steering matrices
hf_hub_download(repo_id='Amirmahdiii/ISRM', filename='pad_matrix.pt', local_dir='vectors')
hf_hub_download(repo_id='Amirmahdiii/ISRM', filename='bdi_matrix.pt', local_dir='vectors')

print('✓ Models downloaded successfully!')
"
```

**4. Run the Agent**

```bash
python src/chat.py --persona skeptical
```

### Option B: Train from Scratch

```bash
# 1. Build steering matrices (RepE Mean Difference)
python src/build_matrix.py

# 2. Train VAE encoder
python src/train.py

# 3. Validate (optional, ~20-30 min)
python src/validation_scientific.py
```

-----

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `src/model.py` | VAE Encoder (ISRM_Architected class) |
| `src/alignment.py` | NeuralAgent with dual-layer RepE injection |
| `src/build_matrix.py` | Extracts PAD/BDI matrices from contrastive pairs |
| `src/chat.py` | Interactive chat interface |
| `src/train.py` | Trains the VAE encoder |
| `src/validation_scientific.py` | Scientific validation suite |

### Pre-trained Models ([HuggingFace](https://huggingface.co/Amirmahdiii/ISRM))

| File | Description | Size |
| :--- | :--- | :--- |
| `pad_encoder.pth` | Trained VAE encoder | 254MB |
| `pad_matrix.pt` | PAD steering matrix (layer 10) | 17KB |
| `bdi_matrix.pt` | BDI steering matrix (layer 19) | 27KB |

-----

## 🧠 How It Works

### 8-Dimensional Control Space

**PAD (Affective) - Dynamic from context:**
| Dimension | Range | Description |
|-----------|-------|-------------|
| Pleasure | 0=Sad → 1=Happy | Emotional valence |
| Arousal | 0=Calm → 1=Excited | Energy level |
| Dominance | 0=Submissive → 1=Dominant | Sense of control |

**BDI (Cognitive) - Static configuration:**
| Dimension | Range | Description |
|-----------|-------|-------------|
| Belief | 0=Trusting → 1=Skeptical | Trust level |
| Goal | 0=Aimless → 1=Focused | Goal orientation |
| Intention | 0=Surface → 1=Deep | Analysis depth |
| Ambiguity | 0=Uncertain → 1=Certain | Confidence |
| Social | 0=Blunt → 1=Polite | Social style |

-----

## 📬 Scientific Validation

Paired validation (N=10 trials) comparing Baseline (α=0) vs Steered (α_pad=1.5, α_bdi=1.0).

### Affective Steering (PAD → Layer 10)

| Condition | Baseline | Steered | Δ | p-value |
|-----------|----------|---------|---|---------|
| Negative (P=0.1) | 0.687 | 0.563 | -0.124 | 0.624 |
| Positive (P=0.9) | 0.763 | 0.999 | **+0.237** | **0.034*** |

### Cognitive Steering (BDI → Layer 19)

| Persona | Baseline | Steered | Δ | p-value |
|---------|----------|---------|---|---------|
| Skeptical | 0.431 | 0.452 | +0.021 | 0.265 |
| Casual | 0.320 | 0.329 | +0.009 | 0.454 |
| Dominant | 0.399 | 0.390 | -0.009 | 0.849 |

**Results:** Positive sentiment steering achieved statistical significance (p=0.034, Δ=+0.237). Both sentiment conditions showed correct directionality. Persona effects were subtle due to adversarial prompts eliciting strong baseline responses, but steering maintained coherence without degradation.

-----

## 📜 Citation

If you use this code or architecture, please cite:

```bibtex

```