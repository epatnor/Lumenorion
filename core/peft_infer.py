# peft_infer.py

# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Modell- och adapterinställningar
BASE_MODEL = "google/gemma-7b-it"
LORA_PATH = "./output_gemma_lora"

# Ladda tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Ladda basmodell och applicera LoRA-adapter
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# Generera svar baserat på prompt
def generate_reply(prompt, max_new_tokens=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
