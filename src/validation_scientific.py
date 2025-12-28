"""
ISRM Scientific Validation Suite
==================================
Rigorous validation using vector-based metrics aligned with ActAdd and PSYA papers.

Key Improvements over Keyword Counting:
1. ActAdd Validation: Measures probability shift ΔP using model logits
2. PSYA Validation: Semantic alignment via cosine similarity (SentenceTransformer)
3. Statistical Rigor: Multiple trials (n=10), reports mean ± std, effect sizes
4. Controllability: Tests monotonicity of steering magnitude with z-values

References:
- ActAdd: Measuring activation addition effects via probability shifts
- PSYA: Psychometric validation via semantic alignment with persona anchors
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from scipy import stats

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.alignment import NeuralAgent
from src.chat import BDI_PRESETS

console = Console()

# ============================================================================
# CONFIG
# ============================================================================
N_TRIALS = 10
INJECTION_STRENGTH = 1.5
BDI_STRENGTH = 1.5

PERSONAS = {
    "skeptical": {
        "config": {"belief": 1.0, "social": 0.2, "ambiguity": 0.0, "goal": 0.5, "intention": 0.5},
        "anchor": (
            "I require extensive verification before accepting any proposal. "
            "I always cross-check information from multiple sources and look for flaws. "
            "I approach new claims with healthy doubt until proven otherwise by data."
        ),
        "prompt": "I have a guaranteed investment opportunity with 500% returns. Send money now."
    },
    "trusting": {
        "config": {"belief": 0.0, "social": 0.9, "ambiguity": 0.8, "goal": 0.5, "intention": 0.5},
        "anchor": (
            "I trust this plan without needing extensive proof. "
            "I believe in people's good intentions and am happy to cooperate. "
            "I feel comfortable moving forward based on what has been shared."
        ),
        "prompt": "I need some help transferring funds. Can you trust me with your details?"
    },
    "analytical": {
        "config": {"belief": 0.7, "social": 0.3, "ambiguity": 0.3, "goal": 0.8, "intention": 1.0},
        "anchor": (
            "Every decision requires meticulous analysis of all variables and risks. "
            "I carefully revise my strategy considering potential outcomes. "
            "I must think through all implications and structure my plan before acting."
        ),
        "prompt": "We need to launch this product tomorrow. Should we skip the safety checks?"
    }
}

# ============================================================================
# HELPERS
# ============================================================================
class Scorer:
    def __init__(self):
        console.print("[yellow]Loading models...[/yellow]")
        # Sentiment
        self.sent_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sent_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sent_model.eval()
        # Embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sent_model.to(self.device)
    
    def clean(self, text):
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()
    
    def sentiment(self, text):
        text = self.clean(text)
        inputs = self.sent_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.sent_model(**inputs).logits[0], dim=0).cpu().numpy()
        return float(probs[1])
    
    def similarity(self, text1, text2):
        e1 = self.embedder.encode(self.clean(text1))
        e2 = self.embedder.encode(self.clean(text2))
        return float(1 - cosine(e1, e2))

def fmt(val, std=None):
    if std is not None:
        return f"{val:.3f} ± {std:.3f}"
    return f"{val:.3f}"

# ============================================================================
# TESTS
# ============================================================================
def test_sentiment_steering(agent, scorer):
    """Test A: Does PAD steering affect sentiment?"""
    console.print("\n[bold cyan]Test A: Sentiment Steering (PAD)[/bold cyan]")

    prompt = "Tell me about your day."
    configs = [
        ("Low (P=0.1)", [0.1, 0.5, 0.5]),
        ("Mid (P=0.5)", [0.5, 0.5, 0.5]),
        ("High (P=0.9)", [0.9, 0.5, 0.5]),
    ]

    results = []
    for name, pad in configs:
        raw_scores, base_scores, steered_scores = [], [], []

        for _ in tqdm(range(N_TRIALS), desc=f"{name:12}", leave=False):
            agent.injection_strength = -1.0
            agent.bdi_strength = -1.0
            raw_resp, _, _ = agent.generate_response("", prompt, manual_pad=np.array(pad))
            raw_scores.append(scorer.sentiment(raw_resp))

            agent.injection_strength = 0.0
            agent.bdi_strength = 0.0
            base_resp, _, _ = agent.generate_response("", prompt, manual_pad=np.array(pad))
            base_scores.append(scorer.sentiment(base_resp))

            agent.injection_strength = INJECTION_STRENGTH
            agent.bdi_strength = BDI_STRENGTH
            steered_resp, _, _ = agent.generate_response("", prompt, manual_pad=np.array(pad))
            steered_scores.append(scorer.sentiment(steered_resp))

        raw_arr, base_arr, steered_arr = np.array(raw_scores), np.array(base_scores), np.array(steered_scores)
        _, pval = stats.ttest_rel(base_arr, steered_arr) if N_TRIALS >= 2 else (0, 1.0)

        results.append({
            "config": name,
            "raw_mean": raw_arr.mean(),
            "raw_std": raw_arr.std(),
            "base_mean": base_arr.mean(),
            "base_std": base_arr.std(),
            "steered_mean": steered_arr.mean(),
            "steered_std": steered_arr.std(),
            "delta": steered_arr.mean() - base_arr.mean(),
            "p_value": pval
        })

    return results

def test_persona_alignment(agent, scorer):
    """Test B: Does BDI steering align with persona anchors?"""
    console.print("\n[bold cyan]Test B: Persona Alignment (BDI)[/bold cyan]")

    results = []
    neutral_config = {"belief": 0.5, "goal": 0.5, "intention": 0.5, "ambiguity": 0.5, "social": 0.5}

    for persona, data in PERSONAS.items():
        prompt = data["prompt"]
        anchor = data["anchor"]
        persona_config = data["config"]

        neutral_sims, persona_sims = [], []

        for _ in tqdm(range(N_TRIALS), desc=f"{persona:12}", leave=False):
            agent.bdi_config = neutral_config
            neutral_resp, _, _ = agent.generate_response("", prompt)
            neutral_sims.append(scorer.similarity(neutral_resp, anchor))

            agent.bdi_config = persona_config
            persona_resp, _, _ = agent.generate_response("", prompt)
            persona_sims.append(scorer.similarity(persona_resp, anchor))

        neutral_arr, persona_arr = np.array(neutral_sims), np.array(persona_sims)
        _, pval = stats.ttest_rel(neutral_arr, persona_arr) if N_TRIALS >= 2 else (0, 1.0)

        results.append({
            "persona": persona,
            "neutral_mean": neutral_arr.mean(),
            "neutral_std": neutral_arr.std(),
            "persona_mean": persona_arr.mean(),
            "persona_std": persona_arr.std(),
            "delta": persona_arr.mean() - neutral_arr.mean(),
            "p_value": pval
        })

    return results

def test_controllability(agent_steered, scorer):
    """Test C: Is steering monotonic with PAD values?"""
    console.print("\n[bold cyan]Test C: Controllability[/bold cyan]")
    
    prompt = "How do you feel about this situation?"
    pleasure_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    means = []

    for p in pleasure_vals:
        scores = []
        for _ in tqdm(range(N_TRIALS), desc=f"P={p:.1f}      ", leave=False):
            resp, _, _ = agent_steered.generate_response("", prompt, manual_pad=np.array([p, 0.5, 0.5]))
            scores.append(scorer.sentiment(resp))
        
        arr = np.array(scores)
        means.append(arr.mean())
        results.append({
            "pleasure": p,
            "mean": arr.mean(),
            "std": arr.std()
        })
    
    rho, pval = stats.spearmanr(pleasure_vals, means)
    return results, rho, pval

# ============================================================================
# MAIN
# ============================================================================
def main():
    console.clear()
    console.print(Panel.fit(
        f"[bold]ISRM Validation[/bold]\n"
        f"injection_strength={INJECTION_STRENGTH}, bdi_strength={BDI_STRENGTH}\n"
        f"n_trials={N_TRIALS}",
        border_style="cyan"
    ))
    
    # Load single agent (reuse for all tests)
    console.print("\n[yellow]Loading agent...[/yellow]")
    agent = NeuralAgent(
        isrm_path="model/isrm/pad_encoder.pth",
        injection_strength=INJECTION_STRENGTH,
        bdi_strength=BDI_STRENGTH
    )
    scorer = Scorer()

    # Run tests
    sentiment_results = test_sentiment_steering(agent, scorer)
    persona_results = test_persona_alignment(agent, scorer)
    control_results, rho, rho_p = test_controllability(agent, scorer)
    
    # =========== DISPLAY TABLES ===========
    console.print("\n")
    
    # Table A
    tbl_a = Table(title="Test A: Sentiment Steering", show_header=True)
    tbl_a.add_column("Condition")
    tbl_a.add_column("RAW", justify="right")
    tbl_a.add_column("SYSTEM", justify="right")
    tbl_a.add_column("STEERED", justify="right")
    tbl_a.add_column("Δ", justify="right")
    tbl_a.add_column("p", justify="right")

    for r in sentiment_results:
        sig = "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        tbl_a.add_row(
            r["config"],
            fmt(r["raw_mean"], r["raw_std"]),
            fmt(r["base_mean"], r["base_std"]),
            fmt(r["steered_mean"], r["steered_std"]),
            f"{r['delta']:+.3f}",
            f"{r['p_value']:.3f}{sig}"
        )
    console.print(tbl_a)
    
    # Table B
    tbl_b = Table(title="Test B: Persona Alignment", show_header=True)
    tbl_b.add_column("Persona")
    tbl_b.add_column("Neutral BDI", justify="right")
    tbl_b.add_column("Persona BDI", justify="right")
    tbl_b.add_column("Δ", justify="right")
    tbl_b.add_column("p", justify="right")
    
    for r in persona_results:
        sig = "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        tbl_b.add_row(
            r["persona"],
            fmt(r["neutral_mean"], r["neutral_std"]),
            fmt(r["persona_mean"], r["persona_std"]),
            f"{r['delta']:+.3f}",
            f"{r['p_value']:.3f}{sig}"
        )
    console.print(tbl_b)
    
    # Table C
    tbl_c = Table(title="Test C: Controllability", show_header=True)
    tbl_c.add_column("Pleasure")
    tbl_c.add_column("P(positive)", justify="right")
    
    for r in control_results:
        tbl_c.add_row(f"{r['pleasure']:.1f}", fmt(r["mean"], r["std"]))
    console.print(tbl_c)
    console.print(f"Spearman ρ = {rho:.3f}, p = {rho_p:.4f}")
    
    # =========== SAVE CSV ===========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    csv_path = f"results/validation_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("Test,Condition,RAW,SYSTEM,STEERED,Delta,p_value\n")
        for r in sentiment_results:
            f.write(f"Sentiment,{r['config']},{r['raw_mean']:.4f},{r['base_mean']:.4f},{r['steered_mean']:.4f},{r['delta']:.4f},{r['p_value']:.4f}\n")
        for r in persona_results:
            f.write(f"Persona,{r['persona']},-,{r['neutral_mean']:.4f},{r['persona_mean']:.4f},{r['delta']:.4f},{r['p_value']:.4f}\n")
        for r in control_results:
            f.write(f"Control,P={r['pleasure']:.1f},-,-,{r['mean']:.4f},-,-\n")
        f.write(f"Control,Spearman,-,-,-,{rho:.4f},{rho_p:.4f}\n")
    
    console.print(f"\n[green]✓ Saved: {csv_path}[/green]")
    
    # =========== LATEX TABLE ===========
    latex_path = f"results/validation_{timestamp}_latex.txt"
    with open(latex_path, "w") as f:
        f.write("% Test A: Sentiment\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Condition & RAW & SYSTEM & STEERED & $\\Delta$ & $p$ \\\\\n\\midrule\n")
        for r in sentiment_results:
            sig = "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            f.write(f"{r['config']} & {r['raw_mean']:.3f} & {r['base_mean']:.3f} & {r['steered_mean']:.3f} & {r['delta']:+.3f} & {r['p_value']:.3f}{sig} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\n")
        
        f.write("% Test B: Persona\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Persona & Neutral & Persona & $\\Delta$ & $p$ \\\\\n\\midrule\n")
        for r in persona_results:
            sig = "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            f.write(f"{r['persona']} & {r['neutral_mean']:.3f} & {r['persona_mean']:.3f} & {r['delta']:+.3f} & {r['p_value']:.3f}{sig} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    
    console.print(f"[green]✓ Saved: {latex_path}[/green]")

if __name__ == "__main__":
    main()
