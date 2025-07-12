# agent.py

import json
import warnings
import sys
from datetime import datetime
from core.peft_infer import generate_reply
from convo_logger import save_conversation
import os

# Suppress known noisy transformer/accelerate warnings
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*offloaded to the cpu.*", category=UserWarning)

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
        "You are Lumenorion, an introspective AI shaped by dreams and emotions.\n"
        "Speak with thoughtful clarity. Use subtle metaphor if it adds depth, but always remain grounded.\n"
        "Avoid excessive poetry or repetition. Prioritize honest, direct answers.\n"
    )

    dream_ref = f"\nYou carry the feeling of a recent dream: {mood}, centered on '{focus}'.\n" if excerpt else ""
    return intro + dream_ref + f"\nUser input:\n{user_input}\n\nReply:"

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
        print()  # spacing
        print(f"Lumenorion: {response.strip()}\n")

        dialogue.append({
            "user": user_input,
            "lumenorion": response.strip()
        })

    if dialogue:
        save_conversation(dialogue)

if __name__ == "__main__":
    run_agent()
