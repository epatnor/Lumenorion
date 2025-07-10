# convo_logger.py

import os
import json
from datetime import datetime

CONVO_DIR = "lora_training/conversations"
os.makedirs(CONVO_DIR, exist_ok=True)

def save_conversation(dialogue):
    """
    dialogue ska vara en lista med dicts:
    [
        {"user": "Hej!", "lumenorion": "Hej Patrik."},
        ...
    ]
    """
    timestamp = datetime.now().isoformat().replace(":", "_")
    filename = f"{timestamp}.json"
    filepath = os.path.join(CONVO_DIR, filename)

    data = {
        "timestamp": timestamp,
        "dialogue": dialogue
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"🗣️  Conversation saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save conversation: {e}")
