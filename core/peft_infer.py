# peft_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from config import LORA_DIR, TEMPERATURE, TOP_P, MAX_TOKENS

# 🧠 Ladda LoRA-konfiguration och basmodell
peft_config = PeftConfig.from_pretrained(LORA_DIR)
base_model_name = peft_config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 🪄 Applicera LoRA-adapter
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

# 🗨️ Generera svar från prompt
def generate_reply(prompt, max_new_tokens=None, temperature=TEMPERATURE):
    max_new_tokens = max_new_tokens or MAX_TOKENS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔧 Tokenisera input och flytta till rätt enhet
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)

    # 🚀 Generera text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.1
        )

    # ✂️ Extrahera endast svaret (trunkera bort prompten)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded[len(prompt):].strip()
    return reply
