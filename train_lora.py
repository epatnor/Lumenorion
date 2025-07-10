# train_lora.py

import os
import subprocess
import sys

DATA_SCRIPT = "prepare_lora_data.py"
MODELFILE_PATH = "Modelfile"
MODEL_NAME = "lumenorion-lora"

def run_prepare_script():
    print("🔄 Preparing dataset...")
    result = subprocess.run([sys.executable, DATA_SCRIPT], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to prepare dataset:")
        print(result.stderr)
        sys.exit(1)
    print("✅ Dataset ready.")

def verify_modelfile():
    if not os.path.exists(MODELFILE_PATH):
        print(f"❌ Modelfile not found: {MODELFILE_PATH}")
        sys.exit(1)

def run_ollama_create():
    print(f"🚀 Training LoRA model: {MODEL_NAME}")
    result = subprocess.run(["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to train LoRA model:")
        print(result.stderr)
        sys.exit(1)
    print("🎉 LoRA model created successfully!")

if __name__ == "__main__":
    run_prepare_script()
    verify_modelfile()
    run_ollama_create()
