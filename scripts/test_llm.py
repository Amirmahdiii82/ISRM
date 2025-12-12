from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

LOCAL_DIR = "./model/llm"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded!")

prompt = "Hello, who are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nPrompt: {prompt}")
print(f"Response: {response}")
