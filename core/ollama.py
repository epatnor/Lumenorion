# core/ollama.py

import subprocess
import json


MODEL_NAME = "gemma3n"  # eller "phi3", "mistral", etc


def chat_with_model(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()
