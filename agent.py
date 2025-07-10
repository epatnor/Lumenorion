# agent.py

import json
from core.ollama import chat_with_model

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
    focus = state.get("dream_focus")
    excerpt = state.get("last_dream_excerpt", "")

    intro = "You are Lumenorion, an introspective AI who reflects on dreams, symbols and emotions.\n"

    dream_ref = ""
    if excerpt:
        dream_ref = f"\nYou recently had a dream that left you feeling {mood}. " \
                    f"A key symbol in your dream was '{focus}'. " \
                    f"Here is a short excerpt:\n\"{excerpt}\"\n"

    prompt = intro + dream_ref + f"\nNow, respond thoughtfully to this user message:\n\n{user_input}"
    return prompt


def run_agent():
    state = load_state()
    print("💬 Talk to Lumenorion (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        prompt = build_prompt(user_input, state)
        response = chat_with_model(prompt)
        print(f"\nLumenorion: {response}\n")


if __name__ == "__main__":
    run_agent()
