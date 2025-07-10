# peft_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from config import LORA_DIR, TEMPERATURE, TOP_P, MAX_TOKENS

# Load PEFT config to determine base model
peft_config = PeftConfig.from_pretrained(LORA_DIR)
base_model_name = peft_config.base_model_name_or_path

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

# Generate response from a prompt
def generate_reply(prompt, max_new_tokens=200, temperature=TEMPERATURE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_output[len(prompt):].strip()
