# export_lora_dataset.py

import os
import json
from datetime import datetime

# Var datan ligger
BASE_DIR = "lora_training"
DREAMS_DIR = os.path.join(BASE_DIR, "dreams")
REFLECT_DIR = os.path.join(BASE_DIR, "reflections")
CONVO_DIR = os.path.join(BASE_DIR, "conversations")
OUTPUT_FILE = os.path.join(BASE_DIR, "datasets", "lumenorion_lora.jsonl")


def ensure_dirs():
    for d in [DREAMS_DIR, REFLECT_DIR, CONVO_DIR, os.path.dirname(OUTPUT_FILE)]:
        os.makedirs(d, exist_ok=True)


def load_json_files(folder):
    data = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                try:
                    data.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping corrupt file: {fname}")
    return data


def extract_from_dreams():
    entries = []
    dreams = load_json_files(DREAMS_DIR)
    for dream in dreams:
        input_text = f"Prompt: {dream.get('prompt', '')}"
        output_text = f"Dream: {dream.get('text', '')}"
        entries.append({"input": input_text.strip(), "output": output_text.strip()})
    return entries


def extract_from_reflections():
    entries = []
    reflections = load_json_files(REFLECT_DIR)
    for ref in reflections:
        input_text = f"Dream symbols: {', '.join(ref.get('symbols', []))}. Mood: {ref.get('mood', '')}"
        output_text = ref.get("reflection", "")
        if output_text:
            entries.append({"input": input_text.strip(), "output": output_text.strip()})
    return entries


def extract_from_conversations():
    entries = []
    convos = load_json_files(CONVO_DIR)
    for convo in convos:
        for exchange in convo.get("dialogue", []):
            user = exchange.get("user")
            lumen = exchange.get("lumenorion")
            if user and lumen:
                entries.append({
                    "input": f"User: {user.strip()}",
                    "output": f"Lumenorion: {lumen.strip()}"
                })
    return entries


def export_dataset():
    ensure_dirs()
    data = extract_from_dreams() + extract_from_reflections() + extract_from_conversations()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Exported {len(data)} training entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    export_dataset()
