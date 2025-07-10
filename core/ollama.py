# core/ollama.py

import subprocess
import json

# == Modellinställningar ==
BASE_MODEL = "gemma3n"           # T.ex. "gemma3n", "mistral", "phi3"
LORA_NAME = "lumenorion-lora"    # Din finetunade LoRA (lägg till .mod om det behövs)
USE_LORA = True                  # Slå på/av LoRA här

def get_model_name():
    return f"{BASE_MODEL}:{LORA_NAME}" if USE_LORA else BASE_MODEL


def chat_with_model(prompt):
    try:
        model_name = get_model_name()

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
