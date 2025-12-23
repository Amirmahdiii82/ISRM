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

    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507", target_layer=-8):
        """
        Args:
            model_name: HuggingFace model ID or local path
            target_layer: Layer to extract activations from (negative = from end)
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

        self.target_layer = target_layer
        self.activations = {}

    def _register_hook(self, layer_idx):
        """Register forward hook to capture activations"""
        def hook_fn(module, input, output):
            # For transformer models, output is typically (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Take the last token's hidden state (where generation continues)
            self.activations['current'] = hidden_states[:, -1, :].detach()

        # Access the correct layer
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

    def build_steering_matrix(self, contrastive_pairs_path="dataset/contrastive_pairs.json"):
        """
        Build steering matrix from contrastive pairs using Mean Difference.

        Returns:
            steering_matrix: Tensor of shape (8, hidden_dim)
        """
        # Load contrastive pairs
        with open(contrastive_pairs_path, 'r') as f:
            data = json.load(f)

        pairs = data['pairs']
        num_dims = data['metadata']['num_dimensions']

        print(f"Loaded {len(pairs)} contrastive pairs for {num_dims} dimensions")

        # Register hook on target layer
        handle = self._register_hook(self.target_layer)

        # Extract steering vectors for each dimension
        steering_vectors = []

        for dim_idx in range(num_dims):
            dim_pairs = [p for p in pairs if p['dimension_idx'] == dim_idx]
            dim_name = dim_pairs[0]['dimension_name']

            print(f"\nProcessing dimension {dim_idx}: {dim_name}")

            pole_a_activations = []
            pole_b_activations = []

            for pair in tqdm(dim_pairs, desc=f"Extracting {dim_name}"):
                # Extract activations for both poles
                act_a = self.extract_activation(pair['pole_a_text'])
                act_b = self.extract_activation(pair['pole_b_text'])

                pole_a_activations.append(act_a)
                pole_b_activations.append(act_b)

            # Stack and compute mean
            pole_a_mean = torch.stack(pole_a_activations).mean(dim=0).squeeze()
            pole_b_mean = torch.stack(pole_b_activations).mean(dim=0).squeeze()

            # Steering vector = difference between poles
            steering_vector = pole_a_mean - pole_b_mean

            # Normalize (optional but recommended)
            steering_vector = steering_vector / (steering_vector.norm() + 1e-8)

            steering_vectors.append(steering_vector)

            print(f"  Vector shape: {steering_vector.shape}")
            print(f"  Vector norm: {steering_vector.norm():.4f}")

        # Remove hook
        handle.remove()

        # Stack into matrix (8, hidden_dim)
        steering_matrix = torch.stack(steering_vectors)

        print(f"\nSteering Matrix Shape: {steering_matrix.shape}")

        return steering_matrix

    def save_steering_matrix(self, matrix, output_path="vectors/steering_matrix.pt"):
        """Save steering matrix to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.save({
            'steering_matrix': matrix,
            'layer_idx': self.target_layer,
            'hidden_dim': matrix.shape[1],
            'num_dimensions': matrix.shape[0],
        }, output_path)

        print(f"\nSaved steering matrix to: {output_path}")
        print(f"Matrix shape: {matrix.shape}")


def main():
    print("=" * 60)
    print("Building Steering Matrix with RepE")
    print("=" * 60)

    extractor = SteeringVectorExtractor()

    print("\nExtracting steering vectors...")
    matrix = extractor.build_steering_matrix()

    print("\nSaving matrix...")
    extractor.save_steering_matrix(matrix)

    print("\n" + "=" * 60)
    print("Done! Steering matrix ready for injection.")
    print("=" * 60)


if __name__ == "__main__":
    main()
