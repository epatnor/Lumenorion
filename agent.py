# agent.py

import json
import warnings
import sys
from datetime import datetime
from core.peft_infer import generate_reply
from convo_logger import save_conversation
from memory import init_db  # framtidssäkrat, körs inte om inte kallat
import os

# Suppress known noisy warnings from transformers / accelerate
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*offloaded to the cpu.*", category=UserWarning)

STATE_PATH = "state.json"

# Load state from JSON (for mood, dream focus etc.)
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

# Build prompt from user input and saved dream state
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

    dream_ref = f"\nYou carry the feeling of a recent dream: {mood}, centered on '{focus}'.\n" if excerpt else ""
    return intro + dream_ref + f"\nUser input:\n{user_input}\n\nReply:"

# Optional future hook — auto-memories (not yet wired)
def auto_save_to_memory(dialogue):
    # Placeholder — optionally parse/summarize memory-worthy statements
    pass

# Main dialogue loop
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
        # Optionally save memory later
        # auto_save_to_memory(dialogue)

if __name__ == "__main__":
    run_agent()
