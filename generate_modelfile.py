# generate_modelfile_clean.py
with open("Modelfile", "w", encoding="utf-8", newline="\n") as f:
    f.write("""FROM gemma3n
ADAPTER lora
PARAMETER r 8
PARAMETER alpha 16
PARAMETER dropout 0.05
PARAMETER bias none
TRAIN_DATA lora_training/datasets/lumenorion_lora_shuffled.jsonl
TEMPLATE {{ input }}\\n{{ output }}
""")

print("✅ Clean Modelfile written.")
