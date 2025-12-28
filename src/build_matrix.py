"""
Build Steering Matrix using Representation Engineering (RepE)
Extracts steering vectors from contrastive pairs using Mean Difference method
"""
import json
import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class SteeringVectorExtractor:
    """Extracts steering vectors from a frozen LLM using contrastive pairs"""

    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507"):
        """
        Args:
            model_name: HuggingFace model ID or local path
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load model path
        if os.path.exists("./model/llm"):
            model_path = "./model/llm"
            print(f"Loading local LLM from {model_path}")
        else:
            model_path = model_name
            print(f"Loading LLM from HuggingFace: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model (frozen for RepE)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model.eval()

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.activations = {}

        # Print model info for verification
        num_layers = len(self.model.model.layers)
        print(f"\n[CONFIG] Model has {num_layers} layers")
        print(f"[CONFIG] PAD extraction: layer 10 (~31% depth)")
        print(f"[CONFIG] BDI extraction: layer 19 (~59% depth)\n")

    def _register_hook(self, layer_idx):
        """Register forward hook to capture activations"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations['current'] = hidden_states[:, -1, :].detach()

        layer = self.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        return handle

    def extract_activation(self, text):
        """Extract activation for a single text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True)

        return self.activations['current'].cpu()

    def build_steering_matrices(self, contrastive_pairs_path="dataset/contrastive_pairs.json"):
        """
        Build separate PAD and BDI steering matrices from contrastive pairs.

        Returns:
            pad_matrix: Tensor of shape (3, hidden_dim) from layer 10
            bdi_matrix: Tensor of shape (5, hidden_dim) from layer 19
        """
        with open(contrastive_pairs_path, 'r') as f:
            data = json.load(f)

        pairs = data['pairs']
        num_dims = data['metadata']['num_dimensions']

        print(f"Loaded {len(pairs)} contrastive pairs for {num_dims} dimensions")

        pad_vectors = []
        bdi_vectors = []
        pad_stats = []
        bdi_stats = []

        for dim_idx in range(num_dims):
            dim_pairs = [p for p in pairs if p['dimension_idx'] == dim_idx]
            dim_name = dim_pairs[0]['dimension_name']

            # PAD: indices 0,1,2 from layer 10
            # BDI: indices 3,4,5,6,7 from layer 19
            is_pad = dim_idx < 3
            target_layer = 10 if is_pad else 19

            print(f"\nProcessing dimension {dim_idx}: {dim_name} (layer {target_layer})")

            handle = self._register_hook(target_layer)

            pole_a_activations = []
            pole_b_activations = []

            for pair in tqdm(dim_pairs, desc=f"Extracting {dim_name}"):
                act_a = self.extract_activation(pair['pole_a_text'])
                act_b = self.extract_activation(pair['pole_b_text'])

                pole_a_activations.append(act_a)
                pole_b_activations.append(act_b)

            pole_a_mean = torch.stack(pole_a_activations).mean(dim=0).squeeze()
            pole_b_mean = torch.stack(pole_b_activations).mean(dim=0).squeeze()

            steering_vector = pole_a_mean - pole_b_mean
            original_norm = steering_vector.norm().item()

            if is_pad:
                pad_vectors.append(steering_vector)
                pad_stats.append({'dim': dim_name, 'norm': original_norm})
            else:
                bdi_vectors.append(steering_vector)
                bdi_stats.append({'dim': dim_name, 'norm': original_norm})

            print(f"  Vector norm: {original_norm:.4f}")
            handle.remove()

        pad_matrix = torch.stack(pad_vectors)
        bdi_matrix = torch.stack(bdi_vectors)

        print("\n" + "=" * 60)
        print("STEERING MATRICES DIAGNOSTICS")
        print("=" * 60)
        print(f"PAD Matrix: {pad_matrix.shape} (layer 10)")
        for stat in pad_stats:
            print(f"  [{stat['dim']:12}] norm={stat['norm']:.4f}")
        print(f"\nBDI Matrix: {bdi_matrix.shape} (layer 19)")
        for stat in bdi_stats:
            print(f"  [{stat['dim']:12}] norm={stat['norm']:.4f}")
        print("=" * 60)

        return pad_matrix, bdi_matrix

    def save_steering_matrices(self, pad_matrix, bdi_matrix):
        """Save PAD and BDI matrices to separate files"""
        os.makedirs("vectors", exist_ok=True)

        torch.save({
            'matrix': pad_matrix,
            'layer': 10,
            'dimensions': ["pleasure", "arousal", "dominance"]
        }, "vectors/pad_matrix.pt")

        torch.save({
            'matrix': bdi_matrix,
            'layer': 19,
            'dimensions': ["belief", "goal", "intention", "ambiguity", "social"]
        }, "vectors/bdi_matrix.pt")

        print(f"\nSaved PAD matrix to: vectors/pad_matrix.pt")
        print(f"  Shape: {pad_matrix.shape}, Layer: 10")
        print(f"Saved BDI matrix to: vectors/bdi_matrix.pt")
        print(f"  Shape: {bdi_matrix.shape}, Layer: 19")


def main():
    print("=" * 60)
    print("Building Steering Matrices with RepE (Dual-Layer)")
    print("=" * 60)

    extractor = SteeringVectorExtractor()

    print("\nExtracting steering vectors...")
    pad_matrix, bdi_matrix = extractor.build_steering_matrices()

    print("\nSaving matrices...")
    extractor.save_steering_matrices(pad_matrix, bdi_matrix)

    print("\n" + "=" * 60)
    print("Done! Steering matrices ready for injection.")
    print("PAD (affective) → Layer 10")
    print("BDI (cognitive) → Layer 19")
    print("=" * 60)


if __name__ == "__main__":
    main()
