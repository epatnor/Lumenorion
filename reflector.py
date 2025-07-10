# reflector.py

import os
import sqlite3
import json
from datetime import datetime
from core.peft_infer import generate_reply

DB_PATH = "lumenorion.db"
LO_RA_REFLECT_DIR = "lora_training/reflections"
REFLECT_LOG_DIR = "lora_training/logs_reflect"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                dream_id INTEGER,
                symbols TEXT,
                mood TEXT,
                reflection TEXT
            )
        """)
        conn.commit()
    print("✅ Reflections table ready.")

def get_latest_dream():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, symbols, text FROM dreams ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
    if row:
        return {"id": row[0], "symbols": row[1].split(","), "text": row[2]}
    return None

def reflect_on_latest_dream():
    init_db()
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
    try:
        reflection = generate_reply(prompt)
    except Exception as e:
        print(f"❌ Failed to generate reflection: {e}")
        return

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
    os.makedirs(LO_RA_REFLECT_DIR, exist_ok=True)
    filename = f"{timestamp.replace(':', '_')}.json"

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO reflections (timestamp, dream_id, symbols, mood, reflection)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, dream_id, ",".join(symbols), mood, reflection.strip()))
        conn.commit()

    data = {
        "timestamp": timestamp,
        "dream_id": dream_id,
        "symbols": symbols,
        "mood": mood,
        "reflection": reflection.strip()
    }

    try:
        with open(os.path.join(LO_RA_REFLECT_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Reflection saved to DB and {filename}")
    except Exception as e:
        print(f"❌ Failed to save reflection to file: {e}")

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

    try:
        with open(os.path.join(REFLECT_LOG_DIR, f"{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Failed to write log entry: {e}")

if __name__ == "__main__":
    reflect_on_latest_dream()
