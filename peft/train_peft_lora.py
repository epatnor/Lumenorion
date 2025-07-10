# train_peft_lora.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
from config import BASE_MODEL, MAX_TOKENS

# File paths
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "peft/output_gemma_lora"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

# LoRA config
config = LoraConfig(
    r=8,
    alpha=16,
    dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
lora_model = get_peft_model(model, config)

# Load and tokenize dataset
train_ds = load_dataset("json", data_files=DATA_PATH)["train"]

def tokenize(example):
    return tokenizer(
        example["input"] + "\n" + example["output"],
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length"
    )

train_ds = train_ds.map(tokenize, batched=True)

# Training parameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=1,
    logging_steps=10,
)

# Train model
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds
)
trainer.train()

# Save LoRA adapter
lora_model.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA trained and saved in {OUTPUT_DIR}")
