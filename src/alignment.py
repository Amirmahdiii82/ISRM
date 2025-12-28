"""
RepE Neural Agent using Representation Engineering.

Architecture:
- Dynamic PAD (3D): Encoder predicts [Pleasure, Arousal, Dominance]
- Static BDI (5D): Configured via bdi_config [Belief, Goal, Intention, Ambiguity, Social]

Dual-Layer Injection:
  PAD → Layer 10 (~31% depth): v_pad = z_pad @ M_pad
  BDI → Layer 19 (~59% depth): v_bdi = bdi_static @ M_bdi
  No signal interference - independent injections
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import ISRM_Architected
import os

# Default BDI configuration (neutral values)
DEFAULT_BDI = {
    "belief": 0.5,      
    "goal": 0.5,        
    "intention": 0.5,   
    "ambiguity": 0.5,  
    "social": 0.7       
}


class NeuralAgent(nn.Module):
    def __init__(
        self,
        isrm_path="./model/isrm/pad_encoder.pth",
        llm_model_name="Qwen/Qwen3-4B-Thinking-2507",
        steering_matrix_path="vectors/steering_matrix.pt",
        injection_layer=16,
        injection_strength=0.2,
        bdi_config=None,
        bdi_strength=1.0
    ):
        """
        Args:
            isrm_path: Path to PAD encoder weights (3D output)
            llm_model_name: HuggingFace model ID or local path
            steering_matrix_path: Deprecated (kept for backward compatibility)
            injection_layer: Deprecated (kept for backward compatibility)
            injection_strength: Global scaling for PAD injection vector
            bdi_config: Dict with keys [belief, goal, intention, ambiguity, social]
                        Each value in [0, 1]. Uses DEFAULT_BDI if None.
            bdi_strength: Scaling factor for BDI component
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bdi_config = bdi_config if bdi_config else DEFAULT_BDI.copy()
        self.bdi_strength = bdi_strength
        self._validate_bdi_config()

        self.isrm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.isrm = ISRM_Architected("distilbert-base-uncased", latent_dim=3).to(self.device)
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
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(self.device)
        self.llm.eval()

        # Steering configuration
        self.pad_layer = 10
        self.bdi_layer = 19
        self.injection_strength = injection_strength

        # Load separate PAD and BDI matrices
        self._load_steering_matrices()

        self.pad_hook_handle = None
        self.bdi_hook_handle = None
        self.pad_injection_vector = None
        self.bdi_injection_vector = None

        # Dimension names
        self.pad_names = ["pleasure", "arousal", "dominance"]
        self.bdi_names = ["belief", "goal", "intention", "ambiguity", "social"]

    def _validate_bdi_config(self):
        """Validate BDI config has all required keys with valid values."""
        required = ["belief", "goal", "intention", "ambiguity", "social"]
        for key in required:
            if key not in self.bdi_config:
                raise ValueError(f"bdi_config missing required key: {key}")
            val = self.bdi_config[key]
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"bdi_config['{key}'] must be in [0, 1], got {val}")

    def _load_steering_matrices(self):
        """Load separate PAD and BDI steering matrices."""
        pad_path = "vectors/pad_matrix.pt"
        bdi_path = "vectors/bdi_matrix.pt"

        if os.path.exists(pad_path) and os.path.exists(bdi_path):
            print(f"Loading PAD matrix from {pad_path}...")
            pad_data = torch.load(pad_path, map_location=self.device)
            self.M_pad = pad_data['matrix'].to(self.device)

            print(f"Loading BDI matrix from {bdi_path}...")
            bdi_data = torch.load(bdi_path, map_location=self.device)
            self.M_bdi = bdi_data['matrix'].to(self.device)

            print(f"  M_pad: {self.M_pad.shape} → layer {self.pad_layer}")
            print(f"  M_bdi: {self.M_bdi.shape} → layer {self.bdi_layer}")
            print(f"  BDI config: {self.bdi_config}")
        else:
            raise ValueError(f"Matrices not found. Run build_matrix.py first.")

    def get_pad_state(self, context):
        """Get PAD vector from encoder (3D affective state)."""
        inputs = self.isrm_tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            z, _, _ = self.isrm(inputs['input_ids'], inputs['attention_mask'])

        return z.cpu().numpy()[0]  

    def get_bdi_vector(self):
        """Get static BDI vector from config (5D cognitive state)."""
        return [
            self.bdi_config["belief"],
            self.bdi_config["goal"],
            self.bdi_config["intention"],
            self.bdi_config["ambiguity"],
            self.bdi_config["social"]
        ]

    def compute_injection_vectors(self, z_pad):
        """
        Compute separate PAD and BDI injection vectors.

        Args:
            z_pad: numpy array of shape (3,) with values in [0, 1]
                   [pleasure, arousal, dominance]

        Returns:
            v_pad: torch.Tensor of shape (hidden_dim,)
            v_bdi: torch.Tensor of shape (hidden_dim,)
            components: dict with norms for diagnostics
        """
        # PAD component (dynamic)
        z_pad_tensor = torch.tensor(z_pad, dtype=self.M_pad.dtype, device=self.device)
        z_pad_normalized = (z_pad_tensor - 0.5) * 2
        v_pad = z_pad_normalized @ self.M_pad
        v_pad = v_pad * self.injection_strength

        # BDI component (static)
        bdi_values = torch.tensor(self.get_bdi_vector(), dtype=self.M_bdi.dtype, device=self.device)
        bdi_normalized = (bdi_values - 0.5) * 2
        v_bdi = bdi_normalized @ self.M_bdi
        v_bdi = v_bdi * self.bdi_strength

        components = {
            "v_pad_norm": v_pad.norm().item(),
            "v_bdi_norm": v_bdi.norm().item()
        }

        return v_pad.squeeze(), v_bdi.squeeze(), components

    def _create_pad_hook(self):
        """Create forward hook for PAD injection at layer 10"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if self.pad_injection_vector is not None:
                hidden_states = hidden_states + self.pad_injection_vector.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook_fn

    def _create_bdi_hook(self):
        """Create forward hook for BDI injection at layer 19"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if self.bdi_injection_vector is not None:
                hidden_states = hidden_states + self.bdi_injection_vector.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook_fn

    def generate_response(self, history_text, user_input, manual_pad=None):
        """
        Generate response with hybrid RepE injection.

        Args:
            history_text: Conversation history string
            user_input: Current user message
            manual_pad: Optional manual PAD override (3D array: [pleasure, arousal, dominance])

        Returns:
            response: Generated text
            injection_info: Dict with injection metadata
            state_info: Dict with PAD (dynamic) and BDI (static) values
        """
        # 1. Get PAD vector (dynamic from encoder or manual override)
        if manual_pad is not None:
            z_pad = manual_pad
        else:
            # Format MUST match training: "Context: {history} \n User: {message}"
            full_context = f"Context: {history_text[-200:]} \n User: {user_input}"
            z_pad = self.get_pad_state(full_context)

        # 2. Compute separate injection vectors
        self.pad_injection_vector, self.bdi_injection_vector, components = self.compute_injection_vectors(z_pad)

        # 3. Register forward hooks on separate layers
        pad_target_layer = self.llm.model.layers[self.pad_layer]
        self.pad_hook_handle = pad_target_layer.register_forward_hook(
            self._create_pad_hook()
        )

        bdi_target_layer = self.llm.model.layers[self.bdi_layer]
        self.bdi_hook_handle = bdi_target_layer.register_forward_hook(
            self._create_bdi_hook()
        )

        # 4. Prepare prompt
        messages = [{"role": "user", "content": user_input}]
        text_prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.llm_tokenizer(text_prompt, return_tensors="pt").to(self.device)

        # 5. Generate
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

        # 6. Cleanup
        if self.pad_hook_handle is not None:
            self.pad_hook_handle.remove()
            self.pad_hook_handle = None
        if self.bdi_hook_handle is not None:
            self.bdi_hook_handle.remove()
            self.bdi_hook_handle = None

        # 7. Prepare metadata
        injection_info = {
            "pad_layer": self.pad_layer,
            "bdi_layer": self.bdi_layer,
            "injection_strength": self.injection_strength,
            "bdi_strength": self.bdi_strength,
            **components
        }

        state_info = {
            "pad": {name: float(z_pad[i]) for i, name in enumerate(self.pad_names)},
            "bdi": self.bdi_config.copy()
        }

        return response, injection_info, state_info

    def set_bdi(self, **kwargs):
        """Update BDI config at runtime. Only updates provided keys."""
        for key, val in kwargs.items():
            if key in self.bdi_config:
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{key} must be in [0, 1], got {val}")
                self.bdi_config[key] = val


# Backward compatibility
Agent_OpenEnded = NeuralAgent