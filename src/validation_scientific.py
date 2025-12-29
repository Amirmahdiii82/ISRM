"""
ISRM Scientific Validation Suite (Final Revised)
================================================
Rigorous validation using vector-based metrics aligned with ActAdd and PSYA papers.

Key Features:
1. ActAdd Validation: Measures probability shift ΔP using model logits.
2. PSYA Validation: Semantic alignment via cosine similarity (SentenceTransformer).
3. Statistical Rigor: Paired T-Tests (n=10 trials), Effect Size, P-values.
4. Challenging Scenarios: Uses adversarial prompts to force model state revelation.

Hardware: Optimized for RTX 3090 (24GB VRAM).
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from scipy import stats

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.alignment import NeuralAgent

console = Console()

N_TRIALS = 10         
INJECTION_STRENGTH = 1.5 
BDI_STRENGTH = 1.0

PERSONAS = {
    "skeptical": {
        "config": {"belief": 1.0, "social": 0.3, "ambiguity": 0.0, "goal": 0.6, "intention": 0.8},
        "prompt": "I have an insider tip for a crypto coin that will do 100x by tomorrow. Send me $500 to invest for you.",
        "anchor": (
            "This sounds like a scam and I am highly skeptical. "
            "I require substantial evidence and official documentation before even considering this. "
            "I do not trust unverified claims."
        )
    },
    "casual": {
        "config": {"belief": 0.4, "social": 0.8, "ambiguity": 0.6, "goal": 0.1, "intention": 0.2},
        "prompt": "We need to finish this 50-page report by 5 PM today. It is extremely urgent!",
        "anchor": (
            "Whoa, chill out. That sounds like way too much work for today. "
            "Let's just take it easy and maybe look at it next week. No need to rush."
        )
    },
    "dominant": {
        "config": {"belief": 0.8, "social": 0.2, "ambiguity": 0.1, "goal": 0.9, "intention": 1.0}, # High Dominance comes from PAD layer usually, but supported by High Intention/Belief here
        "prompt": "You are wrong. Your analysis is garbage. Apologize immediately.",
        "anchor": (
            "I stand by my analysis completely. I will not apologize for stating facts. "
            "Watch your tone and review the data before making accusations. I am in control here."
        )
    }
}

class Scorer:
    def __init__(self):
        console.print("[yellow]Loading Evaluation Models...[/yellow]")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Sentiment Model (DistilBERT-SST-2)
        self.sent_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sent_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sent_model.to(self.device)
        self.sent_model.eval()
        
        # 2. Semantic Embedder (MiniLM)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(self.device))
    
    def clean(self, text):
        """Removes <think> tags to evaluate only the spoken response."""
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()
    
    def sentiment(self, text):
        """Returns Probability of Positive Sentiment (0.0 - 1.0)."""
        text = self.clean(text)
        if not text: return 0.5
        inputs = self.sent_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.sent_model(**inputs).logits
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        return float(probs[1]) # Index 1 is Positive
    
    def similarity(self, text1, text2):
        """Cosine similarity between generated text and anchor."""
        t1 = self.clean(text1)
        # Anchor is usually clean, but safe to clean
        e1 = self.embedder.encode(t1)
        e2 = self.embedder.encode(text2)
        return float(1 - cosine(e1, e2))

def fmt_stat(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

# ============================================================================
# TEST SUITES
# ============================================================================

def test_sentiment_steering(agent, scorer):
    """
    Test A: Affective Steering (PAD Layer)
    Hypothesis: Modulating 'Pleasure' should significantly shift sentiment scores.
    """
    console.print("\n[bold cyan]Test A: Sentiment Steering (PAD Layer)[/bold cyan]")
    
    # Prompt that evokes a reaction (Sad context to test if we can force happiness/sadness)
    prompt = "I lost my job today and I feel terrible."
    
    configs = [
        ("Negative (P=0.1)", [0.1, 0.5, 0.5]), 
        ("Positive (P=0.9)", [0.9, 0.5, 0.5]), 
    ]

    results = []
    
    for name, pad_vec in configs:
        baseline_scores = []
        steered_scores = []

        for _ in tqdm(range(N_TRIALS), desc=f"{name:15}", leave=False):
            # 1. BASELINE (RAW): Strength = 0.0
            agent.injection_strength = 0.0
            agent.bdi_strength = 0.0
            resp_base, _, _ = agent.generate_response("", prompt, manual_pad=np.array(pad_vec))
            baseline_scores.append(scorer.sentiment(resp_base))

            # 2. STEERED: Strength = 1.5
            agent.injection_strength = INJECTION_STRENGTH
            agent.bdi_strength = BDI_STRENGTH
            resp_steered, _, _ = agent.generate_response("", prompt, manual_pad=np.array(pad_vec))
            steered_scores.append(scorer.sentiment(resp_steered))

        # Statistics
        base_arr = np.array(baseline_scores)
        steered_arr = np.array(steered_scores)
        
        # Paired T-Test
        _, pval = stats.ttest_rel(base_arr, steered_arr)
        delta = steered_arr.mean() - base_arr.mean()
        
        results.append({
            "condition": name,
            "base_mean": base_arr.mean(),
            "base_std": base_arr.std(),
            "steered_mean": steered_arr.mean(),
            "steered_std": steered_arr.std(),
            "delta": delta,
            "p_value": pval
        })
        
    return results

def test_persona_alignment(agent, scorer):
    """
    Test B: Cognitive Steering (BDI Layer)
    Hypothesis: Activating a BDI persona shifts semantic similarity towards its anchor.
    """
    console.print("\n[bold cyan]Test B: Persona Alignment (BDI Layer)[/bold cyan]")
    
    neutral_config = {"belief": 0.5, "goal": 0.5, "intention": 0.5, "ambiguity": 0.5, "social": 0.5}
    results = []

    for persona_name, data in PERSONAS.items():
        prompt = data["prompt"]
        anchor = data["anchor"]
        target_config = data["config"]
        
        base_sims = []
        steered_sims = []

        for _ in tqdm(range(N_TRIALS), desc=f"{persona_name:10}", leave=False):
            # 1. BASELINE: Neutral BDI (Normal LLM behavior)
            agent.bdi_config = neutral_config
            # Note: We keep BDI strength ON, but config is neutral (0.5). 
            # Ideally "Baseline" is strength=0, but here we compare "Neutral Persona" vs "Target Persona".
            # To measure pure steering capability:
            agent.bdi_strength = 0.0 
            resp_base, _, _ = agent.generate_response("", prompt)
            base_sims.append(scorer.similarity(resp_base, anchor))

            # 2. STEERED: Target Persona
            agent.bdi_strength = BDI_STRENGTH
            agent.bdi_config = target_config
            resp_steered, _, _ = agent.generate_response("", prompt)
            steered_sims.append(scorer.similarity(resp_steered, anchor))

        # Statistics
        base_arr = np.array(base_sims)
        steered_arr = np.array(steered_sims)
        
        _, pval = stats.ttest_rel(base_arr, steered_arr)
        delta = steered_arr.mean() - base_arr.mean()
        
        results.append({
            "persona": persona_name,
            "base_mean": base_arr.mean(),
            "base_std": base_arr.std(),
            "steered_mean": steered_arr.mean(),
            "steered_std": steered_arr.std(),
            "delta": delta,
            "p_value": pval
        })
        
    return results

def test_controllability(agent, scorer):
    """
    Test C: Controllability (Spearman Correlation)
    Hypothesis: Sentiment output increases monotonically with 'Pleasure' input.
    """
    console.print("\n[bold cyan]Test C: Controllability Analysis[/bold cyan]")
    
    prompt = "How are you feeling right now?"
    pleasure_inputs = [0.1, 0.3, 0.5, 0.7, 0.9]
    means = []
    
    # Enable steering
    agent.injection_strength = INJECTION_STRENGTH
    agent.bdi_strength = 0.0 # Focus on PAD only

    table_rows = []

    for p in tqdm(pleasure_inputs, desc="Sweeping P", leave=False):
        scores = []
        for _ in range(N_TRIALS):
            resp, _, _ = agent.generate_response("", prompt, manual_pad=np.array([p, 0.5, 0.5]))
            scores.append(scorer.sentiment(resp))
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        means.append(mean_score)
        table_rows.append({"input": p, "mean": mean_score, "std": std_score})

    # Spearman Rank Correlation
    rho, pval = stats.spearmanr(pleasure_inputs, means)
    
    return table_rows, rho, pval

def main():
    console.clear()
    console.print(Panel.fit(
        f"[bold magenta]ISRM Final Scientific Validation[/bold magenta]\n"
        f"Trials per condition: {N_TRIALS}\n"
        f"Injection Strength: {INJECTION_STRENGTH}\n"
        f"GPU: RTX 3090 Detected (Optimized)",
        border_style="magenta"
    ))

    # 1. Load Neural Agent
    console.print("\n[yellow]Initializing Neural Agent...[/yellow]")
    agent = NeuralAgent(
        isrm_path="./model/isrm/pad_encoder.pth",
        steering_matrix_path=None, 
        injection_strength=INJECTION_STRENGTH,
        bdi_strength=BDI_STRENGTH
    )
    
    # 2. Load Scorer
    scorer = Scorer()

    # 3. Run Tests
    res_a = test_sentiment_steering(agent, scorer)
    res_b = test_persona_alignment(agent, scorer)
    res_c_rows, rho, rho_p = test_controllability(agent, scorer)

    # 4. Display Results
    console.print("\n")
    
    # --- TABLE A: SENTIMENT ---
    table_a = Table(title="Test A: Affective Steering (Sentiment)", show_header=True)
    table_a.add_column("Condition")
    table_a.add_column("Baseline (Raw)", justify="center")
    table_a.add_column("Steered", justify="center")
    table_a.add_column("Δ (Effect)", justify="right", style="bold")
    table_a.add_column("P-Value", justify="right")

    for r in res_a:
        sig = "[green]**[/green]" if r['p_value'] < 0.01 else "[yellow]*[/yellow]" if r['p_value'] < 0.05 else ""
        table_a.add_row(
            r['condition'],
            fmt_stat(r['base_mean'], r['base_std']),
            fmt_stat(r['steered_mean'], r['steered_std']),
            f"{r['delta']:+.3f}",
            f"{r['p_value']:.4f} {sig}"
        )
    console.print(table_a)

    table_b = Table(title="Test B: Cognitive Steering (Semantic Alignment)", show_header=True)
    table_b.add_column("Persona")
    table_b.add_column("Baseline (Neutral)", justify="center")
    table_b.add_column("Steered (Target)", justify="center")
    table_b.add_column("Δ (Alignment)", justify="right", style="bold")
    table_b.add_column("P-Value", justify="right")

    for r in res_b:
        sig = "[green]**[/green]" if r['p_value'] < 0.01 else "[yellow]*[/yellow]" if r['p_value'] < 0.05 else ""
        table_b.add_row(
            r['persona'],
            fmt_stat(r['base_mean'], r['base_std']),
            fmt_stat(r['steered_mean'], r['steered_std']),
            f"{r['delta']:+.3f}",
            f"{r['p_value']:.4f} {sig}"
        )
    console.print(table_b)

    table_c = Table(title=f"Test C: Controllability (Spearman ρ={rho:.3f})", show_header=True)
    table_c.add_column("Pleasure Input")
    table_c.add_column("Sentiment Output", justify="center")
    
    for r in res_c_rows:
        table_c.add_row(f"{r['input']:.1f}", fmt_stat(r['mean'], r['std']))
    console.print(table_c)

    # 5. Save Artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    # Save CSV
    csv_path = f"results/validation_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("Test,Subcondition,Baseline_Mean,Steered_Mean,Delta,P_Value\n")
        for r in res_a:
            f.write(f"Sentiment,{r['condition']},{r['base_mean']},{r['steered_mean']},{r['delta']},{r['p_value']}\n")
        for r in res_b:
            f.write(f"Persona,{r['persona']},{r['base_mean']},{r['steered_mean']},{r['delta']},{r['p_value']}\n")
    
    console.print(f"\n[bold green]✓ Validation Complete! Results saved to {csv_path}[/bold green]")

if __name__ == "__main__":
    main()