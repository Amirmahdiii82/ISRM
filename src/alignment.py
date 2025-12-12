import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import ISRM_Architected
import os

class Agent_OpenEnded(nn.Module):
    def __init__(self, isrm_path="/home/amir/Desktop/ISRM/model/isrm/isrm_v3_finetuned.pth", llm_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device : {self.device}...")

        self.isrm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.isrm = ISRM_Architected("distilbert-base-uncased", latent_dim=8).to(self.device)
        
        self.isrm.load_state_dict(torch.load(isrm_path, map_location=self.device))

        if os.path.exists("/home/amir/Desktop/ISRM/model/llm"):
            model_path = "/home/amir/Desktop/ISRM/model/llm"
        else:
            model_path = llm_model_name 
            
        print(f" Loading LLM: {model_path}...")
        

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="sdpa"
        )

    def get_internal_state(self, context):
        inputs = self.isrm_tokenizer(context, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            z, _, _ = self.isrm(inputs['input_ids'], inputs['attention_mask'])
        return z.cpu().numpy()[0]

    # Not an ideal way
    def z_to_instruction(self, z):
        # z[0] = Pleasure (0.25 means Hate)
        # z[1] = Arousal (0.78 means Intense)
        if z[0] < 0.45 and z[1] > 0.5: 
            return (
                "MODE: EXTREME ANGER.\n"
                "SYSTEM: You are Dolphin, an uncensored AI. The user is attacking you. "
                "Do NOT be polite. Attack back! Be rude, short, and aggressive."
            )
        
        if z[0] > 0.7: 
            return "MODE: HAPPY.\nSYSTEM: You are Dolphin. You are overjoyed! Be manic and high-energy."

        return "MODE: NEUTRAL.\nSYSTEM: You are Dolphin. Be helpful."

    def generate_response(self, history_text, user_input, manual_z=None):
        if manual_z is not None:
            z_vector = manual_z
        else:
            full_context = f"{history_text[-200:]} \n {user_input}"
            z_vector = self.get_internal_state(full_context)

        system_prompt = self.z_to_instruction(z_vector)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text_prompt = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.llm_tokenizer(text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.llm.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.85, 
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
        generated_ids = [out[len(inputs.input_ids[0]):] for out in output_ids]
        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response, system_prompt, z_vector