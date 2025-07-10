# train_lora.py

import json
import random
import os

INPUT_FILE = "lora_training/datasets/lumenorion_lora.jsonl"
OUTPUT_FILE = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    lines = [json.loads(line) for line in infile]

random.shuffle(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for entry in lines:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"✅ Exported {len(lines)} entries to {OUTPUT_FILE}")
