# train_peft_lora.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import sys, os
import gc
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_MODEL, MAX_TOKENS


DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "peft/output_lora_lumenorion"
CACHE_DIR = "models/gemma3n"


def detect_safe_batch_size():
    return 1  # Hardcoded to be minimal


if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(0.8)
    except Exception:
        pass

gc.collect()
torch.cuda.empty_cache()


print("📦 Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=CACHE_DIR,
    device_map=None,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print(f"✅ Model loaded on: {next(model.parameters()).device}")


print("⚙️  Applying lightweight LoRA config...")
config = LoraConfig(
    r=4,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
lora_model = get_peft_model(model, config)
print("✅ LoRA model wrapped.")


print("📝 Loading dataset...")
train_ds = load_dataset("json", data_files=DATA_PATH)["train"]
train_ds = train_ds.select(range(min(10, len(train_ds))))  # Limit for light training
print(f"📊 Using {len(train_ds)} training examples")

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

print("✍️  Tokenizing...")
train_ds = train_ds.map(tokenize, batched=True, num_proc=1)
print("✅ Tokenization done.")


print("🧠 Sanity check...")
print(f"  Device: {next(lora_model.parameters()).device}")
print(f"  Sample: {train_ds[0]['input_ids'][:10]}")


batch_size = detect_safe_batch_size()
print(f"🚦 Batch size: {batch_size}, Accumulation: 1, Epochs: 1")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    save_steps=1000,
    logging_steps=10,
    report_to="none",
    disable_tqdm=False,
    logging_dir="lora_training/logs_train/tensorboard"
)

print("🚀 Starting training...")
try:
    gc.collect()
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds
    )
    trainer.train()
    print("✅ Training complete.")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("⚠️ CUDA OOM – retrying on CPU...")
        torch.cuda.empty_cache()
        gc.collect()

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            cache_dir=CACHE_DIR,
            device_map=None,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32
        )
        lora_model = get_peft_model(model, config).to("cpu")

        cpu_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=False,
            no_cuda=True,
            save_total_limit=1,
            save_steps=1000,
            logging_steps=10,
            report_to="none",
            disable_tqdm=False,
            logging_dir="lora_training/logs_train/tensorboard"
        )

        trainer = Trainer(
            model=lora_model,
            args=cpu_args,
            train_dataset=train_ds
        )
        trainer.train()
        print("✅ Training complete on CPU.")
    else:
        print("❌ Training failed:")
        traceback.print_exc()
        exit(1)


print(f"💾 Saving LoRA adapter to: {OUTPUT_DIR}")
lora_model.save_pretrained(OUTPUT_DIR)
print("✅ Lightweight LoRA saved.")
