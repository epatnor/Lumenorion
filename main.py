# main.py

from memory import init_db
from dream_engine import generate_dream


if __name__ == "__main__":
    init_db()
    generate_dream()
