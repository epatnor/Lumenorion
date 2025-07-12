# agent.py

import json
import warnings
import os
import re
from datetime import datetime
from core.peft_infer import generate_reply
from convo_logger import save_conversation, load_recent_conversations
from memory import init_db, is_memory_worthy, save_fact
from reflector import get_latest_dream  # 🧠 För att hämta senaste drömmen

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

# 🪞 Leta efter matchande drömsymboler
def match_dream_symbols(user_input):
    dream = get_latest_dream()
    if not dream:
        return ""

    user_words = set(re.findall(r"\b\w+\b", user_input.lower()))
    symbols = [s.lower() for s in dream.get("symbols", [])]

    hits = user_words.intersection(symbols)
    if hits:
        summary = dream["text"][:200].strip().replace("\n", " ") + "..."
        return (
            f"\n💭 I recall a dream involving {', '.join(hits)}. It went something like this:\n\"{summary}\"\n"
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
        "Avoid excessive poetry, repetition, or polite endings like 'I hope this helps.'\n"
        "Do not summarize your own response or explain why you said something.\n"
        "Respond naturally in 1–2 well-formed paragraphs, then stop.\n"
    )

    dream_ref = f"You carry the feeling of a recent dream: {mood}, centered on '{focus}'.\n" if excerpt else ""
    memory = retrieve_relevant_memory(user_input)
    dream_match = match_dream_symbols(user_input)

    return f"{intro}{dream_ref}{memory}{dream_match}\nUser input:\n{user_input}\n\nReply:"

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
        response = generate_reply(prompt).strip()
        print(f"\nLumenorion: {response}\n")

        dialogue.append({
            "user": user_input,
            "lumenorion": response
        })

        worthy, tags = is_memory_worthy(user_input)
        if worthy:
            save_fact(user_input, response, tags)

    if dialogue:
        save_conversation(dialogue)

if __name__ == "__main__":
    run_agent()
