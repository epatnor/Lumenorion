# memory.py

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "lumenorion.db"
LO_RA_DREAMS_DIR = "lora_training/dreams"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # -- DREAMS TABLE --
    c.execute("""
        CREATE TABLE IF NOT EXISTS dreams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbols TEXT,
            text TEXT,
            prompt TEXT
        )
    """)

    # Check for missing columns
    c.execute("PRAGMA table_info(dreams)")
    columns = [row[1] for row in c.fetchall()]
    if "symbols" not in columns:
        print("⚙️ Adding missing column: 'symbols'")
        c.execute("ALTER TABLE dreams ADD COLUMN symbols TEXT")
    if "prompt" not in columns:
        print("⚙️ Adding missing column: 'prompt'")
        c.execute("ALTER TABLE dreams ADD COLUMN prompt TEXT")

    # -- CONVERSATIONS TABLE --
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            lumenorion_response TEXT
        )
    """)

    # -- FACTS TABLE --
    c.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            fact TEXT,
            context TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database ready.")

def save_dream(text, symbols, prompt=""):
    timestamp = datetime.now().isoformat()
    symbol_str = ",".join(symbols)

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO dreams (timestamp, symbols, text, prompt)
        VALUES (?, ?, ?, ?)
    """, (timestamp, symbol_str, text, prompt))
    conn.commit()
    conn.close()

    # Save as JSON for LoRA
    os.makedirs(LO_RA_DREAMS_DIR, exist_ok=True)
    filename = f"{timestamp.replace(':', '_')}.json"
    data = {
        "timestamp": timestamp,
        "symbols": symbols,
        "text": text.strip(),
        "prompt": prompt.strip()
    }
    with open(os.path.join(LO_RA_DREAMS_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Dream saved to DB and {filename}")
