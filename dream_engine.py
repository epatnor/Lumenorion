# dream_engine.py

import random
import datetime
import os
import json
from memory import save_dream
from core.ollama import chat_with_model


# Ord att drömma kring – i framtiden kan dessa hämtas dynamiskt
seed_words = [
    "mirror", "telescope", "feather", "clock", "fog", "ladder",
    "whisper", "labyrinth", "ember", "river", "owl", "typewriter"
]

LOGLORA_DIR = "lora_training/dreams"
os.makedirs(LOGLORA_DIR, exist_ok=True)


def generate_dream():
    # Välj 3–5 slumpmässiga ord
    selected = random.sample(seed_words, k=random.randint(3, 5))
    prompt = (
        f"You are dreaming. In your dream, you encounter: {', '.join(selected)}.\n"
        "Describe the dream in vivid and poetic detail. The dream doesn't need to make logical sense."
    )

    print("💤 Generating dream with prompt:")
    print(prompt)

    dream_text = chat_with_model(prompt)

    # Spara i SQLite-minnet
    save_dream(dream_text, selected)

    # Spara som json för framtida LoRA-träning
    ts = datetime.datetime.now().isoformat().replace(":", "_")
    filename = os.path.join(LOGLORA_DIR, f"{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "prompt": prompt.strip(),
            "text": dream_text.strip(),
            "symbols": selected
        }, f, ensure_ascii=False, indent=2)

    print("\n🌙 Dream generated:\n")
    print(dream_text)
