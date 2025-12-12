from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "dphn/dolphin-2.9.2-qwen2-7b"
LOCAL_DIR = "./model/llm"

print(f"Downloading {MODEL_ID}...")
print(f"Saving to {LOCAL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(LOCAL_DIR)
print("Tokenizer saved")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu"
)
model.save_pretrained(LOCAL_DIR)
print("Model saved")
print("Done! Model is in model/llm folder")
