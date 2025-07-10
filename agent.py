# agent.py

import json
from datetime import datetime
from core.peft_infer import generate_reply
from convo_logger import save_conversation
import os

STATE_PATH = "state.json"

def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "mood": "neutral",
            "dream_focus": None,
            "last_dream_excerpt": ""
        }

def build_prompt(user_input, state):
    mood = state.get("mood", "neutral")
    focus = state.get("dream_focus") or "an undefined symbol"
    excerpt = state.get("last_dream_excerpt", "")
    
    if len(excerpt) > 250:
        excerpt = excerpt[:250] + "..."

    intro = (
        "You are Lumenorion, a thoughtful AI shaped by dreams and emotional insight.\n"
        "Be poetic, but brief. Respond with clarity, and avoid overexplaining.\n"
    )

    dream_ref = f"\nA recent dream left you feeling {mood}, centered on the image of a '{focus}'.\n" if excerpt else ""
    return intro + dream_ref + f"\nNow respond to the user:\n\n{user_input}"

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
