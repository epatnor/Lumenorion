# train_peft_lora.py

import os, sys, traceback, torch, logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from config import BASE_MODEL, MAX_TOKENS

# == Loggfix ==
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# == Paths ==
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "peft/output_gemma_lora"
CACHE_DIR = "models/gemma3n"
MAX_EXAMPLES = 20  # Fail-safe för test

print("📦 Loading tokenizer and base model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=CACHE_DIR,
    device_map=None,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

print(f"✅ Model loaded on: {next(model.parameters()).device}")

# == LoRA config ==
print("⚙️  Applying LoRA configuration...")
config = LoraConfig(
    r=4,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
lora_model = get_peft_model(model, config).to(device)
print("✅ LoRA model wrapped.")

# == Dataset ==
print("📝 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"📊 Loaded dataset: {len(dataset)} examples")

dataset = dataset.select(range(min(len(dataset), MAX_EXAMPLES)))
print(f"📉 Trimmed to {len(dataset)} examples for test run")

def tokenize(batch):
    texts = []
    for input_text, output_text in zip(batch["input"], batch["output"]):
        if isinstance(input_text, list):
            input_text = " ".join(input_text)
        if isinstance(output_text, list):
            output_text = " ".join(output_text)
        texts.append(f"{input_text}\n{output_text}")
    return tokenizer(texts, truncation=True, max_length=MAX_TOKENS, padding="max_length")

print("✍️  Tokenizing...")
dataset = dataset.map(tokenize, batched=True, num_proc=1)
print("✅ Tokenization complete.")

# == Training args ==
print("🚦 Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    max_steps=50,  # Failsafe!
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    save_total_limit=1,
    save_steps=25,
    logging_steps=1,
    report_to="none",
    disable_tqdm=False,
    log_level="info",
    logging_dir="lora_training/logs_train/tensorboard"
)

# == Trainer ==
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset
)

# == Run training ==
print("🚀 Starting training...")
try:
    trainer.train()
    print("✅ Training complete.")
except KeyboardInterrupt:
    print("⏹️ Interrupted by user.")
except Exception as e:
    print("❌ Training failed:")
    traceback.print_exc()
    sys.exit(1)

# == Save LoRA model ==
print(f"💾 Saving to: {OUTPUT_DIR}")
lora_model.save_pretrained(OUTPUT_DIR)
print("✅ Done.")
