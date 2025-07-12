# peft_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from config import LORA_DIR, TEMPERATURE, TOP_P, MAX_TOKENS

# ======================
# 🔧 Modellinitialisering
# ======================

# 🧠 Ladda LoRA-konfiguration och identifiera basmodell
peft_config = PeftConfig.from_pretrained(LORA_DIR)
base_model_name = peft_config.base_model_name_or_path

# 🔤 Ladda tokenizer och basmodell
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 🪄 Ladda och applicera LoRA-adapter
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

# ======================
# 💬 Textgenerering
# ======================

def generate_reply(prompt, max_tokens=None, temperature=TEMPERATURE):
    """
    Generera ett svar från modellen givet en prompt.
    :param prompt: Sträng med användarens prompt
    :param max_tokens: Max antal nya tokens att generera (standard tas från config)
    :param temperature: Temperatur för sampling (default från config)
    :return: Modellens svar som sträng
    """
    max_tokens = max_tokens or MAX_TOKENS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔁 Förbered inputs och flytta till device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if next(model.parameters()).device != device:
        model.to(device)

    # 🚀 Generera svar utan gradientspårning
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1
        )

    # ✂️ Plocka bort prompten från outputen
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded[len(prompt):].strip()
    return reply
