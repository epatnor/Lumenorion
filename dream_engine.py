# dream_engine.py

import os
import json
import random
import datetime
import warnings
from memory import save_dream
from core.peft_infer import generate_reply

# 🔕 Tysta störande varningar från transformers/accelerate
warnings.filterwarnings("ignore", category=UserWarning)

# 🌌 Fröord som inspirerar drömmen
SEED_WORDS = [
    "mirror", "telescope", "feather", "clock", "fog", "ladder",
    "whisper", "labyrinth", "ember", "river", "owl", "typewriter"
]

LOGLORA_DIR = "lora_training/dreams"
os.makedirs(LOGLORA_DIR, exist_ok=True)

def generate_dream():
    selected = random.sample(SEED_WORDS, k=random.randint(3, 5))
    prompt = (
        f"You are dreaming. In your dream, you encounter: {', '.join(selected)}.\n"
        "Describe the dream in vivid and poetic detail. The dream doesn't need to make logical sense."
    )

    print("🌙 Generating dream...")
    print("📝 Prompt:")
    print(prompt + "\n")

    try:
        # 🧠 Generera kortare drömtext
        dream_text = generate_reply(prompt, max_tokens=200).strip()
    except Exception as e:
        print(f"❌ Failed to generate dream: {e}")
        return

    # ⏺️ Spara till databas + LoRA
    save_dream(dream_text, selected, prompt)

    # 💾 Spara till fil (för backup/träning)
    timestamp = datetime.datetime.now().isoformat()
    filename = os.path.join(LOGLORA_DIR, f"{timestamp.replace(':', '_')}.json")

    data = {
        "timestamp": timestamp,
        "prompt": prompt,
        "text": dream_text,
        "symbols": selected
    }

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Dream saved to file: {filename}")
    except Exception as e:
        print(f"❌ Failed to save dream to file: {e}")

    print_dream(dream_text)

def print_dream(dream_text):
    print("\n🌌 Dream output:\n")
    print(dream_text)

if __name__ == "__main__":
    generate_dream()
