# dream_engine.py

import random
import datetime
import os
import json
from memory import save_dream
from core.peft_infer import generate_reply

# Fröord för drömmen
seed_words = [
    "mirror", "telescope", "feather", "clock", "fog", "ladder",
    "whisper", "labyrinth", "ember", "river", "owl", "typewriter"
]

# Katalog för att logga drömmar till LoRA-träning
LOGLORA_DIR = "lora_training/dreams"
os.makedirs(LOGLORA_DIR, exist_ok=True)

def generate_dream():
    selected = random.sample(seed_words, k=random.randint(3, 5))
    prompt = (
        f"You are dreaming. In your dream, you encounter: {', '.join(selected)}.\n"
        "Describe the dream in vivid and poetic detail. The dream doesn't need to make logical sense."
    )
    print("💤 Generating dream with prompt:")
    print(prompt)

    try:
        dream_text = generate_reply(prompt)
    except Exception as e:
        print(f"❌ Failed to generate dream: {e}")
        return

    save_dream(dream_text, selected)

    # Tidsstämpel och filnamn
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(LOGLORA_DIR, f"{ts}.json")

    # Spara dröm till disk
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts,
                "prompt": prompt.strip(),
                "text": dream_text.strip(),
                "symbols": selected
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Failed to save dream to file: {e}")
        return

    print_dream(dream_text)

def print_dream(dream_text):
    print("\n🌙 Dream generated:\n")
    print(dream_text)
