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


def load_recent_conversations(limit=50):
    """
    Laddar de senaste konversationerna som dicts med timestamp + dialogue.
    Returnerar en lista sorterad nyast först.
    """
    try:
        files = sorted(
            [f for f in os.listdir(CONVO_DIR) if f.endswith(".json")],
            key=lambda x: os.path.getmtime(os.path.join(CONVO_DIR, x)),
            reverse=True
        )
        conversations = []
        for filename in files[:limit]:
            filepath = os.path.join(CONVO_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations.append(data)
        return conversations
    except Exception as e:
        print(f"❌ Failed to load conversations: {e}")
        return []
