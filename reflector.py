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
    conn = sqlite3.connect(DB_PATH)
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
    c.execute("PRAGMA table_info(reflections)")
    columns = [row[1] for row in c.fetchall()]
    if "symbols" not in columns:
        print("⚙️  Adding missing column: 'symbols'")
        c.execute("ALTER TABLE reflections ADD COLUMN symbols TEXT")
    if "mood" not in columns:
        print("⚙️  Adding missing column: 'mood'")
        c.execute("ALTER TABLE reflections ADD COLUMN mood TEXT")
    if "reflection" not in columns:
        print("⚙️  Adding missing column: 'reflection'")
        c.execute("ALTER TABLE reflections ADD COLUMN reflection TEXT")
    conn.commit()
    conn.close()
    print("✅ Reflections table ready.")

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
    reflection = generate_reply(prompt)
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
    os.makedirs(
