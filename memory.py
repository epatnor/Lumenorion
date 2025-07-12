# memory.py

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "lumenorion.db"
LO_RA_DREAMS_DIR = "lora_training/dreams"

# == Initialize Database and Tables ==
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
    c.execute("PRAGMA table_info(dreams)")
    dream_cols = [row[1] for row in c.fetchall()]
    if "symbols" not in dream_cols:
        print("⚙️ Adding missing column: 'symbols'")
        c.execute("ALTER TABLE dreams ADD COLUMN symbols TEXT")
    if "prompt" not in dream_cols:
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

    # -- FACTS (MEMORIES) TABLE --
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

# == Save a dream to DB and as JSON ==
def save_dream(text, symbols, prompt=""):
    timestamp = datetime.now().isoformat()
    symbol_str = ",".join(symbols)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO dreams (timestamp, symbols, text, prompt)
        VALUES (?, ?, ?, ?)
    """, (timestamp, symbol_str, text.strip(), prompt.strip()))
    conn.commit()
    conn.close()

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

# == Determine if user input contains memory-worthy content ==
def is_memory_worthy(text):
    text_lower = text.lower()

    memory_phrases = [
        # Swedish
        "jag heter", "mitt namn är", "jag är född", "jag fyller år", "jag är från",
        "jag bor i", "min fru", "min son", "min dotter", "jag reste",
        "jag flyttade", "på min semester", "jag började jobba", "jag gillar",
        "jag älskar", "jag hatar", "jag tycker att", "kom ihåg", "påminn mig",
        "notera detta", "du borde minnas", "tidigare sa du", "när vi pratade om",
    
        # English
        "my name is", "i was born", "my birthday is", "i live in", "i'm from",
        "my wife", "my son", "my daughter", "i traveled", "i moved",
        "on my vacation", "i started working", "i like", "i love", "i hate",
        "i think that", "remember this", "remind me", "you should remember",
        "earlier you said", "when we talked about"
    ]


    tags = []
    for phrase in memory_phrases:
        if phrase in text_lower:
            if "fyller år" in phrase or "född" in phrase:
                tags.append("birthday")
            elif "bor" in phrase or "från" in phrase or "flyttade" in phrase:
                tags.append("location")
            elif "älskar" in phrase or "gillar" in phrase or "hatar" in phrase:
                tags.append("preference")
            elif "påminn" in phrase or "kom ihåg" in phrase:
                tags.append("reminder")
            else:
                tags.append("memory")

    return bool(tags), tags

# == Save a memory/fact if worth storing ==
def save_fact(text, context=""):
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO facts (timestamp, fact, context)
        VALUES (?, ?, ?)
    """, (timestamp, text.strip(), context.strip()))
    conn.commit()
    conn.close()

    print(f"🧠 Fact remembered: \"{text.strip()}\"")

