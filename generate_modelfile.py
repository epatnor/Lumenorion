# generate_modelfile.py

modelfile_content = """FROM gemma3n
ADAPTER lora
PARAMETER r 8
PARAMETER alpha 16
PARAMETER dropout 0.05
PARAMETER bias none
TRAIN_DATA lora_training/datasets/lumenorion_lora_shuffled.jsonl
TEMPLATE \"\"\"{{ input }}\\n{{ output }}\"\"\""""

with open("Modelfile", "w", encoding="utf-8", newline="\n") as f:
    f.write(modelfile_content)

print("✅ Modelfile regenerated with proper UTF-8 + LF line endings.")
