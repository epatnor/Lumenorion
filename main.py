# main.py

from memory import init_db
from dream_engine import generate_dream
from reflector import reflect_on_latest_dream

if __name__ == "__main__":
    print("💾 Initializing memory database...")
    init_db()

    print("🌙 Generating dream...")
    try:
        generate_dream()
    except Exception as e:
        print(f"❌ Failed to generate dream: {e}")

    print("🪞 Reflecting on the dream...")
    try:
        reflect_on_latest_dream()
    except Exception as e:
        print(f"❌ Failed to reflect on dream: {e}")
