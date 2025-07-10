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

    # Trimma excerpt för att undvika för långa svar
    if len(excerpt) > 250:
        excerpt = excerpt[:250] + "..."

    intro = (
        "You are Lumenorion, a thoughtful AI shaped by dreams and emotional insight.\n"
        "Be poetic, but brief. Respond with clarity, and avoid overexplaining.\n"
    )

    dream_ref = ""
    if excerpt:
        dream_ref = f"\nA recent dream left you feeling {mood}, centered on the image of a '{focus}'.\n"

    prompt = intro + dream_ref + f"\nNow respond to the user:\n\n{user_input}"
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
