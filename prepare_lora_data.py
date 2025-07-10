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

    # Filtrera bort trasiga entries
    valid_lines = [entry for entry in lines if "input" in entry and "output" in entry]

    if len(valid_lines) != len(lines):
        print(f"⚠️ Filtered out {len(lines) - len(valid_lines)} invalid entries.")

    num_entries = len(valid_lines)
    input_lens = [len(entry["input"]) for entry in valid_lines]
    output_lens = [len(entry["output"]) for entry in valid_lines]

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
    print()

    # Shuffle och spara
    random.shuffle(valid_lines)
    with open(SHUFFLED_PATH, "w", encoding="utf-8") as f:
        for entry in valid_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Shuffled dataset saved to: {os.path.abspath(SHUFFLED_PATH)}")

    # Visa exempel
    print("\n🔎 First 3 entries:")
    for i, entry in enumerate(valid_lines[:3]):
        print(f"\n--- Entry {i+1} ---")
        print("Input:", entry["input"][:300])
        print("Output:", entry["output"][:300])


if __name__ == "__main__":
    analyze_and_shuffle()
