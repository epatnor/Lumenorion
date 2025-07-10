# peft_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Stigar
LORA_PATH = "./peft/output_gemma_lora"

# Ladda PEFT-konfiguration för att hämta basmodell
peft_config = PeftConfig.from_pretrained(LORA_PATH)
base_model_name = peft_config.base_model_name_or_path

# Ladda tokenizer och basmodell
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Applicera LoRA
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# Generera svar
def generate_reply(prompt, max_new_tokens=200, temperature=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_output[len(prompt):].strip()
