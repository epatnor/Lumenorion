# prepare_lora_data.py

import os
import json
import random
from glob import glob
from datetime import datetime

DATA_DIR = "lora_training"
DATASET_PATH = os.path.join(DATA_DIR, "datasets/lumenorion_lora.jsonl")
SHUFFLED_PATH = os.path.join(DATA_DIR, "datasets/lumenorion_lora_shuffled.jsonl")
STATS_PATH = os.path.join(DATA_DIR, "datasets/stats.json")

def collect_entries():
    entries = []

    # Dreams
    for path in glob(os.path.join(DATA_DIR, "dreams", "*.json")):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
            entries.append({
                "input": f"Dream prompt:\n{obj['prompt']}",
                "output": obj["text"].strip()
            })

    # Reflections
    for path in glob(os.path.join(DATA_DIR, "reflections", "*.json")):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
            summary = f"Mood: {obj.get('mood', 'unknown')} / Symbols: {', '.join(obj.get('symbols', []))}"
            entries.append({
                "input": f"Reflect on the dream summary:\n{summary}",
                "output": obj["reflection"].strip()
            })

    # Conversations
    for path in glob(os.path.join(DATA_DIR, "conversations", "*.json")):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
            for turn in obj.get("dialogue", []):
                user = turn.get("user")
                lumenorion = turn.get("lumenorion")
                if user and lumenorion:
                    entries.append({
                        "input": f"User: {user.strip()}",
                        "output": lumenorion.strip()
                    })

    print(f"🧠 Collected {len(entries)} raw data entries.")
    return entries

def save_dataset(entries):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"💾 Dataset written to {DATASET_PATH}")

def analyze_and_shuffle():
    if not os.path.exists(DATASET_PATH):
        print("📦 No existing dataset found, creating one from raw logs...")
        entries = collect_entries()
        save_dataset(entries)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

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

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("📊 Dataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    random.shuffle(valid_lines)
    with open(SHUFFLED_PATH, "w", encoding="utf-8") as f:
        for entry in valid_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Shuffled dataset saved to: {os.path.abspath(SHUFFLED_PATH)}")

    print("\n🔎 First 3 entries:")
    for i, entry in enumerate(valid_lines[:3]):
        print(f"\n--- Entry {i+1} ---")
        print("Input:", entry["input"][:300])
        print("Output:", entry["output"][:300])

if __name__ == "__main__":
    analyze_and_shuffle()
