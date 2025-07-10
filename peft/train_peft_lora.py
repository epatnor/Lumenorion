# train_peft_lora.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

MODEL_ID = "google/gemma-3b-it"
DATA_PATH = "../lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "./output_gemma_lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)

# LoRA-config
config = LoraConfig(r=8, alpha=16, dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
lora_model = get_peft_model(model, config)

# Dataset
train_ds = load_dataset("json", data_files=DATA_PATH)["train"]
def tokenize(ex):
    return tokenizer(ex["input"] + "\n" + ex["output"], truncation=True, max_length=1024, padding="max_length")
train_ds = train_ds.map(tokenize, batched=True)

# Träningsargument
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=1,
    logging_steps=10,
)

# Träna!
trainer = Trainer(model=lora_model, args=args, train_dataset=train_ds)
trainer.train()
lora_model.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA trained and saved in {OUTPUT_DIR}")
