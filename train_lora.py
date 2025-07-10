# train_lora.py

# Trains a LoRA adapter using PEFT + transformers

import os
import subprocess
import sys

# Prepare and shuffle the dataset
def prepare_dataset():
    print("🔄 Preparing dataset...")
    result = subprocess.run([sys.executable, "prepare_lora_data.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to prepare dataset:")
        print(result.stderr)
        sys.exit(1)
    print("✅ Dataset ready.")

# Trigger the LoRA training script
def train_peft_lora():
    print("🚀 Training LoRA model with PEFT...")
    result = subprocess.run([sys.executable, "peft/train_peft_lora.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to train LoRA model:")
        print(result.stderr)
        sys.exit(1)
    print("🎉 LoRA model trained and saved!")

# Main entry point
if __name__ == "__main__":
    prepare_dataset()
    train_peft_lora()
