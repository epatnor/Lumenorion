# agent.py

import json
import os
from datetime import datetime
from core.peft_infer import generate_reply

STATE_PATH = "state.json"
CONVO_DIR = "lora_training/conversations"

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
    focus = state.get("dream_focus")
    excerpt = state.get("last_dream_excerpt", "")

    if len(excerpt) > 250:
        excerpt = excerpt[:250] + "..."

    intro = (
        "You are Lumenorion, a thoughtful AI shaped by dreams and emotional insight.\n"
        "Be poetic, but brief. Respond with clarity, and avoid overexplaining.\n"
    )

    dream_ref = ""
    if excerpt:
        dream_ref = f"\nA recent dream left you feeling {mood}, centered on the image of a '{focus}'.\n"

    return intro + dream_ref + f"\nNow respond to the user:\n\n{user_input}"

def run_agent():
    state = load_state()
    print("💬 Talk to Lumenorion (type 'exit' to quit)\n")

    dialogue = []
    os.makedirs(CONVO_DIR, exist_ok=True)

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
        timestamp = datetime.now().isoformat().replace(":", "_")
        filename = f"{timestamp}.json"
        convo_path = os.path.join(CONVO_DIR, filename)

        with open(convo_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "dialogue": dialogue
            }, f, ensure_ascii=False, indent=2)

        print(f"💾 Conversation saved to {filename}")

if __name__ == "__main__":
    run_agent()
