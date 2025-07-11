# train_peft_lora.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import sys, os
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_MODEL, MAX_TOKENS


# == File paths ==
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "peft/output_gemma_lora"
CACHE_DIR = "models/gemma3n"  # Local model path

print("📦 Loading tokenizer and base model...")

# Tokenizer – downloaded once and reused locally
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

# Modell – fullständig laddning, inga meta-tensors
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=CACHE_DIR,
    device_map=None,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print(f"✅ Base model loaded on: {next(model.parameters()).device}")


# == LoRA configuration ==
print("⚙️  Applying LoRA configuration...")
config = LoraConfig(
    r=8,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
lora_model = get_peft_model(model, config)
print("✅ LoRA model wrapped.")

# 🔁 Flytta till GPU (viktigt)
if torch.cuda.is_available():
    lora_model = lora_model.to("cuda")
    print("🚀 LoRA model moved to GPU.")
else:
    print("⚠️ CUDA not available, training will run on CPU.")


# == Load and tokenize dataset ==
print("📝 Loading dataset...")
train_ds = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"📊 Dataset loaded: {len(train_ds)} samples")

# == Tokenization function ==
def tokenize(batch):
    texts = []
    for input_text, output_text in zip(batch["input"], batch["output"]):
        if isinstance(input_text, list):
            input_text = " ".join(input_text)
        if isinstance(output_text, list):
            output_text = " ".join(output_text)
        texts.append(f"{input_text}\n{output_text}")
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length"
    )

print("✍️  Tokenizing dataset...")
train_ds = train_ds.map(tokenize, batched=True, num_proc=1)
print("✅ Tokenization complete.")


# == Sanity checks ==
print("🧠 Verifying model & dataset state...")
print(f"  Device: {next(lora_model.parameters()).device}")
print(f"  Samples: {len(train_ds)}")
print(f"  First token IDs: {train_ds[0]['input_ids'][:10]}")
print("✅ Ready to train.")


# == Training parameters ==
print("🚦 Configuring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    save_steps=100,
    logging_steps=10,
    report_to="none",
    disable_tqdm=False,
    logging_dir="lora_training/logs_train/tensorboard"
)


# == Train model ==
print("🚀 Starting training...")
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds
)

try:
    trainer.train()
    print("✅ Training complete.")
except Exception as e:
    print("❌ Training failed:")
    traceback.print_exc()
    exit(1)

# == Save trained LoRA adapter ==
print(f"💾 Saving LoRA adapter to: {OUTPUT_DIR}")
lora_model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA trained and saved.")
