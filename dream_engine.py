# dream_engine.py

import random
import datetime
from memory import save_dream
from ollama import chat_with_model  # Du skapar denna i core senare


# Lista med ordkällor – kan bytas ut mot RSS, API eller slumpfiler i framtiden
seed_words = [
    "mirror", "telescope", "feather", "clock", "fog", "ladder",
    "whisper", "labyrinth", "ember", "river", "owl", "typewriter"
]


def generate_dream():
    # Välj 3–5 slumpmässiga ord
    selected = random.sample(seed_words, k=random.randint(3, 5))
    prompt = f"You are dreaming. In your dream, you encounter: {', '.join(selected)}.\n" \
             "Describe the dream in vivid and poetic detail. The dream doesn't need to make logical sense."

    print("💤 Generating dream with prompt:")
    print(prompt)

    # Svar från lokal LLM
    dream_text = chat_with_model(prompt)

    # Spara i minnet
    save_dream(dream_text, selected)

    print("\n🌙 Dream generated:\n")
    print(dream_text)
