# reflector.py

import os
import json
import sqlite3
import warnings
from datetime import datetime
from core.peft_infer import generate_reply

# 🔕 Tysta varningar
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*offloaded to the cpu.*", category=UserWarning)

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
    print("📚 Table 'reflections' is ready.")

def get_latest_dream():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, symbols, text FROM dreams ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
    if row:
        return {
            "id": row[0],
            "symbols": row[1].split(","),
            "text": row[2].strip()
        }
    return None

def extract_mood(text):
    moods = ["hopeful", "melancholic", "confused", "joyful", "anxious", "nostalgic", "neutral"]
    for mood in moods:
        if mood in text.lower():
            return mood
    return "unclear"

def save_reflection(dream_id, symbols, mood, reflection, timestamp):
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
        print(f"💾 Saved reflection to DB and {filename}")
    except Exception as e:
        print(f"❌ Failed to save JSON reflection: {e}")

def log_reflection(prompt, reflection, mood, symbol, timestamp):
    os.makedirs(REFLECT_LOG_DIR, exist_ok=True)
    log_entry = {
        "timestamp": timestamp,
        "prompt": prompt,
        "response": reflection.strip(),
        "mood": mood,
        "focus_symbol": symbol
    }

    try:
        with open(os.path.join(REFLECT_LOG_DIR, f"{timestamp.replace(':', '_')}.json"), "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Failed to write reflection log: {e}")

def reflect_on_latest_dream():
    init_db()
    dream = get_latest_dream()
    if not dream:
        print("⚠️ No dream available to reflect on.")
        return

    excerpt = dream["text"][:300].replace("\n", " ")
    symbols = dream["symbols"]
    focus_symbol = symbols[0] if symbols else "unknown"

    prompt = (
        f"Reflect on the following dream:\n\n\"{excerpt}\"\n\n"
        f"Based on the symbols ({', '.join(symbols)}), what emotional state might the dream convey?\n"
        f"Summarize the mood and key theme in a short paragraph."
    )

    print("🔍 Generating reflection...")
    try:
        reflection = generate_reply(prompt, max_tokens=200).strip()
    except Exception as e:
        print(f"❌ Failed to generate reflection: {e}")
        return

    timestamp = datetime.now().isoformat()
    mood = extract_mood(reflection)

    print(f"🧠 Mood: {mood}")
    print(f"🌌 Focus symbol: {focus_symbol}")
    print(f"\nLumenorion's reflection:\n{reflection}\n")

    save_reflection(dream["id"], symbols, mood, reflection, timestamp)
    log_reflection(prompt, reflection, mood, focus_symbol, timestamp)

if __name__ == "__main__":
    reflect_on_latest_dream()
