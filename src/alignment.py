import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import ISRM_Architected
import os


class NeuralAgent(nn.Module):
    """
    RepE-based Neural Agent using Representation Engineering.
    Injects steering vectors directly into the LLM's forward pass.
    """

    def __init__(
        self,
        isrm_path="/home/amir/Desktop/ISRM/model/isrm/isrm_v3_finetuned.pth",
        llm_model_name="Qwen/Qwen3-4B-Thinking-2507",
        steering_matrix_path="vectors/steering_matrix.pt",
        injection_layer= 14,
        injection_strength=1.0
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.isrm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.isrm = ISRM_Architected("distilbert-base-uncased", latent_dim=8).to(self.device)
        self.isrm.load_state_dict(torch.load(isrm_path, map_location=self.device))
        self.isrm.eval()

        if os.path.exists("./model/llm"):
            model_path = "./model/llm"
        else:
            model_path = llm_model_name

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.llm.eval()

        # Load Steering Matrix
        self.injection_layer = injection_layer
        self.injection_strength = injection_strength

        if os.path.exists(steering_matrix_path):
            print(f"Loading steering matrix from {steering_matrix_path}...")
            checkpoint = torch.load(steering_matrix_path, map_location=self.device)
            self.steering_matrix = checkpoint['steering_matrix'].to(self.device)
            print(f"  Matrix shape: {self.steering_matrix.shape}")
        else:
            print(f"WARNING: Steering matrix not found at {steering_matrix_path}")
            print("  Run 'python src/build_matrix.py' to generate it.")
            print("  Using zero matrix as placeholder.")
            # Placeholder - assumes hidden_dim from model config
            hidden_dim = self.llm.config.hidden_size
            self.steering_matrix = torch.zeros(8, hidden_dim, device=self.device)

        self.hook_handle = None
        self.injection_vector = None

    def get_internal_state(self, context):
        """Get z-vector from ISRM (8D psychological state)"""
        inputs = self.isrm_tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            z, _, _ = self.isrm(inputs['input_ids'], inputs['attention_mask'])

        return z.cpu().numpy()[0]

    def compute_injection_vector(self, z):
        """
        Compute injection vector from z-state and steering matrix.

        Args:
            z: numpy array of shape (8,) with values in [0, 1]

        Returns:
            injection_vector: torch.Tensor of shape (hidden_dim,)
        """
        # Convert z to torch and normalize to [-1, 1] for bipolar steering
        z_tensor = torch.from_numpy(z).to(dtype=self.steering_matrix.dtype, device=self.device)
        z_normalized = (z_tensor - 0.5) * 2  # [0,1] -> [-1, 1]

        # Matrix multiplication: (1, 8) @ (8, hidden_dim) -> (1, hidden_dim)
        injection_vector = z_normalized @ self.steering_matrix

        # Apply strength scaling
        injection_vector = injection_vector * self.injection_strength

        return injection_vector.squeeze()

    def _create_injection_hook(self):
        """Create a forward hook that adds the injection vector to activations"""

        def hook_fn(module, input, output):
            # Output is typically (hidden_states, ) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add injection vector to all tokens' representations
            # Broadcasting: (batch, seq_len, hidden_dim) + (hidden_dim,)
            if self.injection_vector is not None:
                hidden_states = hidden_states + self.injection_vector.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook_fn

    def generate_response(self, history_text, user_input, manual_z=None):
        """
        Generate response with RepE injection.

        Args:
            history_text: Conversation history string
            user_input: Current user message
            manual_z: Optional manual z-vector override

        Returns:
            response: Generated text
            injection_info: Dict with injection metadata
            z_vector: The psychological state vector used
        """
        # 1. Get z-vector from ISRM
        if manual_z is not None:
            z_vector = manual_z
        else:
            full_context = f"{history_text[-200:]}\n{user_input}"
            z_vector = self.get_internal_state(full_context)

        # 2. Compute injection vector
        self.injection_vector = self.compute_injection_vector(z_vector)

        # 3. Register forward hook
        target_layer = self.llm.model.layers[self.injection_layer]
        self.hook_handle = target_layer.register_forward_hook(
            self._create_injection_hook()
        )

        # 4. Prepare prompt (no system prompt needed - steering is neural)
        messages = [
            {"role": "user", "content": user_input}
        ]

        text_prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.llm_tokenizer(text_prompt, return_tensors="pt").to(self.device)

        # 5. Generate with injection
        with torch.no_grad():
            output_ids = self.llm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

        generated_ids = [out[len(inputs.input_ids[0]):] for out in output_ids]
        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 6. Remove hook
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        # 7. Prepare metadata
        injection_info = {
            "layer": self.injection_layer,
            "strength": self.injection_strength,
            "vector_norm": self.injection_vector.norm().item()
        }

        return response, injection_info, z_vector


# Backward compatibility alias
Agent_OpenEnded = NeuralAgent
