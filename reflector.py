# reflector.py

import os
import sqlite3
import json
from datetime import datetime
from core.ollama import chat_with_model

DB_PATH = "lumenorion.db"
LO_RA_REFLECT_DIR = "lora_training/reflections"
REFLECT_LOG_DIR = "lora_training/logs_reflect"


def get_latest_dream():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, symbols, text FROM dreams ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "symbols": row[1].split(","), "text": row[2]}
    return None


def reflect_on_latest_dream():
    dream = get_latest_dream()
    if not dream:
        print("⚠️ No dream found to reflect on.")
        return

    symbols = dream["symbols"]
    excerpt = dream["text"][:300].replace("\n", " ").strip()

    prompt = (
        f"Reflect on the following dream:\n\n\"{excerpt}\"\n\n"
        f"Based on the symbols ({', '.join(symbols)}), what emotional state might the dream convey? "
        f"Summarize the mood and key theme in a short paragraph."
    )

    print("🔍 Reflecting on latest dream...")
    reflection = chat_with_model(prompt)
    mood = extract_mood(reflection)
    focus_symbol = symbols[0] if symbols else "unknown"

    print(f"🧠 Mood: {mood}")
    print(f"🌌 Focus symbol: {focus_symbol}")

    save_reflection(dream["id"], symbols, mood, reflection)
    log_reflection(prompt, reflection, mood, focus_symbol)


def extract_mood(text):
    moods = ["hopeful", "melancholic", "confused", "joyful", "anxious", "nostalgic", "neutral"]
    for mood in moods:
        if mood in text.lower():
            return mood
    return "unclear"


def save_reflection(dream_id, symbols, mood, reflection):
    timestamp = datetime.now().isoformat()

    # Spara till LoRA-reflections som JSON
    os.makedirs(LO_RA_REFLECT_DIR, exist_ok=True)
    filename = f"{timestamp.replace(':', '_')}.json"
    data = {
        "timestamp": timestamp,
        "dream_id": dream_id,
        "symbols": symbols,
        "mood": mood,
        "reflection": reflection.strip()
    }
    with open(os.path.join(LO_RA_REFLECT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Reflection saved to {filename}")


def log_reflection(prompt, reflection, mood, symbol):
    os.makedirs(REFLECT_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().isoformat().replace(":", "_")
    entry = {
        "timestamp": timestamp,
        "prompt": prompt,
        "response": reflection.strip(),
        "mood": mood,
        "focus_symbol": symbol
    }

    with open(os.path.join(REFLECT_LOG_DIR, f"{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    reflect_on_latest_dream()
