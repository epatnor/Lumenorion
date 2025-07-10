# prepare_lora_data.py

import os
import json
import random

DATASET_PATH = "lora_training/datasets/lumenorion_lora.jsonl"
SHUFFLED_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
STATS_PATH = "lora_training/datasets/stats.json"

def analyze_and_shuffle():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    # Grundläggande statistik
    num_entries = len(lines)
    input_lens = [len(entry["input"]) for entry in lines]
    output_lens = [len(entry["output"]) for entry in lines]

    stats = {
        "total_entries": num_entries,
        "avg_input_length": sum(input_lens) / num_entries if num_entries else 0,
        "avg_output_length": sum(output_lens) / num_entries if num_entries else 0,
        "max_input_length": max(input_lens, default=0),
        "max_output_length": max(output_lens, default=0)
    }

    # Spara statistik
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("📊 Dataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Shuffle och spara
    random.shuffle(lines)
    with open(SHUFFLED_PATH, "w", encoding="utf-8") as f:
        for entry in lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Shuffled dataset saved to {SHUFFLED_PATH}")


if __name__ == "__main__":
    analyze_and_shuffle()
