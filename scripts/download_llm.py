from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
LOCAL_DIR = "./model/llm"

print(f"Downloading {MODEL_ID}...")
print(f"Saving to {LOCAL_DIR}")
print("Note: This model should fit on a 24GB GPU")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(LOCAL_DIR)
print("Tokenizer saved")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
model.save_pretrained(LOCAL_DIR)
print("Model saved")
print("Done! Model is in model/llm folder")
