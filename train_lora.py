# train_lora.py

# Trains a LoRA model using PEFT + Transformers, with dataset preparation and logging

import os
import subprocess
import sys
import json
from datetime import datetime

# Log file path
LOG_PATH = "lora_training/logs_train/last_training_log.json"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Run dataset preparation script
def prepare_dataset():
    print("🔄 Preparing dataset...")
    result = subprocess.run([sys.executable, "prepare_lora_data.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to prepare dataset:")
        print(result.stderr)
        return False, result.stderr
    print("✅ Dataset ready.")
    return True, None

# Run LoRA training script
def train_peft_lora():
    print("🚀 Training LoRA model with PEFT...")
    result = subprocess.run([sys.executable, "peft/train_peft_lora.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to train LoRA model:")
        print(result.stderr)
        return False, result.stderr
    print("🎉 LoRA model trained and saved!")
    return True, None

# Save training metadata to JSON log file
def log_training(success, stage, error=None):
    log = {
        "timestamp": datetime.now().isoformat(),
        "model": "gemma-3n (LoRA)",
        "dataset": "lumenorion_lora_shuffled.jsonl",
        "stage": stage,
        "success": success,
        "error": error.strip() if error else None
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"📝 Training log saved to {LOG_PATH}")

# Run both steps and log outcomes
if __name__ == "__main__":
    ok, error = prepare_dataset()
    if not ok:
        log_training(False, "prepare_dataset", error)
        sys.exit(1)

    ok, error = train_peft_lora()
    if not ok:
        log_training(False, "train_peft_lora", error)
        sys.exit(1)

    log_training(True, "done")
