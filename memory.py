# memory.py

import sqlite3
from datetime import datetime


DB_PATH = "lumenorion.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS dreams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            content TEXT,
            keywords TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_dream(text, keywords):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO dreams (timestamp, content, keywords)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), text.strip(), ", ".join(keywords)))
    conn.commit()
    conn.close()
