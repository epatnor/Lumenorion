# reflector.py

import os
import sqlite3
import json
from datetime import datetime
from core.ollama import chat_with_model

DB_PATH = "lumenorion.db"
LO_RA_REFLECT_DIR = "lora_training/reflections"


def get_latest_dream():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, symbols, text FROM dreams ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "symbols": row[1].split(",") if row[1] else [],
            "text": row[2] or ""
        }
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

    reflection = chat_with_model(prompt)
    mood = extract_mood(reflection)

    print("🔍 Reflecting on latest dream...")
    print(f"🧠 Mood: {mood}")
    print(f"🌌 Focus symbol: {symbols[0] if symbols else 'unknown'}")

    save_reflection(dream["id"], symbols, mood, reflection)


def extract_mood(text):
    moods = [
        "hopeful", "melancholic", "confused",
        "joyful", "anxious", "nostalgic", "neutral"
    ]
    lower = text.lower()
    for mood in moods:
        if mood in lower:
            return mood
    return "unclear"


def save_reflection(dream_id, symbols, mood, reflection):
    timestamp = datetime.now().isoformat()
    os.makedirs(LO_RA_REFLECT_DIR, exist_ok=True)

    filename = f"{timestamp.replace(':', '_')}.json"
    filepath = os.path.join(LO_RA_REFLECT_DIR, filename)

    data = {
        "timestamp": timestamp,
        "dream_id": dream_id,
        "symbols": symbols,
        "mood": mood,
        "reflection": reflection.strip()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Reflection saved to {filename}")
