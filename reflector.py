# reflector.py

import sqlite3
import json
import os
from datetime import datetime


DB_PATH = "lumenorion.db"
STATE_PATH = "state.json"


def get_last_dream():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT content, keywords, timestamp FROM dreams ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "content": row[0],
            "keywords": row[1].split(", "),
            "timestamp": row[2]
        }
    return None


def analyze_dream(dream_text, keywords):
    # Enkel heuristik. Kan ersättas med AI-reflektion senare
    mood = "mysterious"
    if "light" in dream_text or "sun" in dream_text:
        mood = "hopeful"
    elif "fog" in dream_text or "shadows" in dream_text:
        mood = "introspective"
    elif "storm" in dream_text or "chaos" in dream_text:
        mood = "turbulent"

    # Använd första starka bild som "focus"
    focus = None
    for word in keywords:
        if word in dream_text:
            focus = word
            break

    return {
        "mood": mood,
        "focus": focus or keywords[0],
        "keywords": keywords
    }


def update_state(dream_data, interpretation):
    state = {
        "last_updated": datetime.now().isoformat(),
        "mood": interpretation["mood"],
        "dream_focus": interpretation["focus"],
        "dream_keywords": interpretation["keywords"],
        "last_dream_excerpt": dream_data["content"][:300].replace("\n", " ") + "..."
    }

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def reflect_on_latest_dream():
    print("🔍 Reflecting on latest dream...")
    dream = get_last_dream()
    if not dream:
        print("⚠️  No dreams found.")
        return

    interpretation = analyze_dream(dream["content"], dream["keywords"])
    update_state(dream, interpretation)

    print(f"🧠 Mood: {interpretation['mood']}")
    print(f"🌌 Focus symbol: {interpretation['focus']}")
