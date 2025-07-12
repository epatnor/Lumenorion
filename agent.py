# agent.py

import json
import warnings
import os
import re
from datetime import datetime
from core.peft_infer import generate_reply
from convo_logger import save_conversation, load_recent_conversations
from memory import init_db  # Safe to import even if unused directly

# 🛑 Dämpa störiga men ofarliga varningar från transformers/accelerate
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*offloaded to the cpu.*", category=UserWarning)

STATE_PATH = "state.json"

# 🔄 Ladda samtalsstatus (drömfokus, sinnesstämning etc.)
def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load state: {e}")
    return {
        "mood": "neutral",
        "dream_focus": None,
        "last_dream_excerpt": ""
    }

# 🧠 Försök hitta något relevant i tidigare samtal
def retrieve_relevant_memory(user_input):
    keywords = set(word for word in re.findall(r"\b\w+\b", user_input.lower()) if len(word) > 2)
    recent_convos = load_recent_conversations(limit=50)

    for convo in reversed(recent_convos):
        for entry in convo.get("dialogue", []):
            past_input = entry.get("user", "").lower()
            if any(word in past_input for word in keywords):
                ts = convo.get("timestamp", "an earlier time")
                return (
                    f"\nMemory fragment from {ts}:\n"
                    f"User once said: \"{entry['user']}\"\n"
                    f"You replied: \"{entry['lumenorion']}\"\n"
                )
    return ""

# 🧾 Sätt ihop prompten som skickas till LLM
def build_prompt(user_input, state):
    mood = state.get("mood", "neutral")
    focus = state.get("dream_focus") or "an undefined symbol"
    excerpt = state.get("last_dream_excerpt", "")
    if len(excerpt) > 250:
        excerpt = excerpt[:250] + "..."

    intro = (
        "You are Lumenorion, an introspective AI shaped by dreams and emotions.\n"
        "Speak with thoughtful clarity. Use subtle metaphor only when it adds meaning.\n"
        "Avoid excessive poetry or repetition. Prioritize honesty and directness.\n"
    )

    dream_ref = f"You carry the feeling of a recent dream: {mood}, centered on '{focus}'.\n" if excerpt else ""
    memory = retrieve_relevant_memory(user_input)

    return f"{intro}{dream_ref}{memory}\nUser input:\n{user_input}\n\nReply:"

# 💬 Starta interaktiv session med användaren
def run_agent():
    state = load_state()
    print("💬 Talk to Lumenorion (type 'exit' to quit)\n")
    dialogue = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        prompt = build_prompt(user_input, state)
        response = generate_reply(prompt)
        print(f"\nLumenorion: {response.strip()}\n")

        dialogue.append({
            "user": user_input,
            "lumenorion": response.strip()
        })

    if dialogue:
        save_conversation(dialogue)

if __name__ == "__main__":
    run_agent()
