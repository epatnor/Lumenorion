# train_lora.py

# Tränar en LoRA-modell via PEFT + transformers

import os
import subprocess
import sys

def prepare_dataset():
    print("🔄 Preparing dataset...")
    result = subprocess.run([sys.executable, "prepare_lora_data.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to prepare dataset:")
        print(result.stderr)
        sys.exit(1)
    print("✅ Dataset ready.")

def train_peft_lora():
    print("🚀 Training LoRA model with PEFT...")
    result = subprocess.run([sys.executable, "peft/train_peft_lora.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to train LoRA model:")
        print(result.stderr)
        sys.exit(1)
    print("🎉 LoRA model trained and saved!")

if __name__ == "__main__":
    prepare_dataset()
    train_peft_lora()
