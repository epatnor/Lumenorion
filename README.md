# 🌌 Lumenorion

Lumenorion is a poetic AI companion that dreams, reflects, and gently stirs thought.

It is not goal-driven — it seeks meaning.

Powered by local LLMs (e.g., Gemma, Phi-3) via Ollama and structured around a memory-reflection-dream loop, Lumenorion is designed to run efficiently on local hardware without costly API calls.

## 🌱 Project Structure
- `dream_engine/` - Generates nightly dreams from random words or stimuli.
- `memory/` - Stores dreams, reflections, and past thoughts in a long-term memory.
- `reflector/` - Looks back on dreams and events to create insight.
- `core/` - Orchestrates loops, agents, and thought patterns.

## 🧠 Features (in progress)
- Autonomous dream generation
- Persistent memory via SQLite or Chroma
- CLI and daily loop triggers
- Self-reflective journaling
- Eventual proactivity ("Hey Patrik, I had a dream...")

---

## 🛠️ Requirements
- Python 3.10+
- Ollama (for running LLMs locally)
- SQLite or ChromaDB
