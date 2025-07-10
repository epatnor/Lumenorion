# generate_modelfile_clean.py

with open("Modelfile", "w", encoding="utf-8", newline="\n") as f:
    f.write(
        "FROM gemma3n\n"
        "ADAPTER lora\n"
        "PARAMETER r 8\n"
        "PARAMETER alpha 16\n"
        "PARAMETER dropout 0.05\n"
        "PARAMETER bias none\n"
        "TRAIN_DATA ./lora_training/datasets/lumenorion_lora_shuffled.jsonl\n"
        "TEMPLATE {{ input }}\\n{{ output }}\n"
    )

print("✅ Clean Modelfile written.")
