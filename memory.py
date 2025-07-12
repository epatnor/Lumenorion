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

    # Skapa tabell för drömmar
    c.execute("""
        CREATE TABLE IF NOT EXISTS dreams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbols TEXT,
            text TEXT,
            prompt TEXT
        )
    """)

    # Skapa tabell för långtidsminnen (om vi vill spara fakta, datum, viktiga dialoger etc)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            category TEXT,
            summary TEXT,
            full_text TEXT
        )
    """)

    # Kontrollera och uppdatera drömtabellen om kolumner saknas
    c.execute("PRAGMA table_info(dreams)")
    dream_columns = [row[1] for row in c.fetchall()]
    if "symbols" not in dream_columns:
        print("⚙️  Adding missing column: 'symbols'")
        c.execute("ALTER TABLE dreams ADD COLUMN symbols TEXT")
    if "prompt" not in dream_columns:
        print("⚙️  Adding missing column: 'prompt'")
        c.execute("ALTER TABLE dreams ADD COLUMN prompt TEXT")

    conn.commit()
    conn.close()
    print("✅ Database ready.")


def save_dream(text, symbols, prompt=""):
    timestamp = datetime.now().isoformat()
    symbol_str = ",".join(symbols)

    # Spara till SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO dreams (timestamp, symbols, text, prompt)
        VALUES (?, ?, ?, ?)
    """, (timestamp, symbol_str, text.strip(), prompt.strip()))
    conn.commit()
    conn.close()

    # Spara till JSON för LoRA-träning
    os.makedirs(LO_RA_DREAMS_DIR, exist_ok=True)
    filename = f"{timestamp.replace(':', '_')}.json"
    json_path = os.path.join(LO_RA_DREAMS_DIR, filename)
    data = {
        "timestamp": timestamp,
        "symbols": symbols,
        "text": text.strip(),
        "prompt": prompt.strip()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Dream saved to DB and {filename}")


def save_memory(summary, full_text, category="general"):
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO memories (timestamp, category, summary, full_text)
        VALUES (?, ?, ?, ?)
    """, (timestamp, category, summary.strip(), full_text.strip()))
    conn.commit()
    conn.close()
    print(f"🧠 Memory saved ({category}) at {timestamp}")
