# core/ollama.py

import subprocess
import os

MODEL_FILE = os.path.join("core", "model.txt")
DEFAULT_MODEL = "gemma3n"

def get_model_name():
    try:
        with open(MODEL_FILE, "r", encoding="utf-8") as f:
            model = f.read().strip()
            return model if model else DEFAULT_MODEL
    except FileNotFoundError:
        return DEFAULT_MODEL


def model_exists_locally(model_name):
    result = subprocess.run(
        ["ollama", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    return model_name in result.stdout.decode("utf-8")


def chat_with_model(prompt):
    try:
        model_name = get_model_name()

        if ":" in model_name and not model_exists_locally(model_name):
            print(f"⚠️ LoRA model '{model_name}' not found locally. Falling back to base model.")
            model_name = model_name.split(":")[0]

        print(f"🧠 Using model: {model_name}")

        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )

        output = result.stdout.decode("utf-8").strip()
        if not output:
            error = result.stderr.decode("utf-8").strip()
            raise RuntimeError(f"Ollama returned no output.\nError: {error}")

        return output

    except Exception as e:
        print(f"❌ Error during Ollama chat: {e}")
        return "⚠️ I encountered an internal error during processing."


# (Ev framtida stöd)
# def chat_with_system_prompt(system_prompt, user_input):
#     full_prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input}"
#     return chat_with_model(full_prompt)
