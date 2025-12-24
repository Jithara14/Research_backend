from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)

print("Gemma loaded successfully")
