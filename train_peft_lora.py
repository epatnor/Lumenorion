# train_peft_lora.py

# == Setup ==
import os, sys, torch, logging, traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import BASE_MODEL, MAX_TOKENS

# == Init paths ==
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# == Config ==
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "lora_training/outputs/gemma3n_lora_test"
CACHE_DIR = "models/gemma3n"
MAX_EXAMPLES = 40
MAX_STEPS = 20
BATCH_SIZE = 2
MAX_TOKENS = 256

print(f"""🚀 train_peft_lora.py started
🧭 Config:
  DATA_PATH:     {DATA_PATH}
  OUTPUT_DIR:    {OUTPUT_DIR}
  CACHE_DIR:     {CACHE_DIR}
  MAX_EXAMPLES:  {MAX_EXAMPLES}
  MAX_STEPS:     {MAX_STEPS}
  BATCH_SIZE:    {BATCH_SIZE}""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")
if device.type == "cpu":
    print("⚠️  Running on CPU – training will be slow.")

# == Load model ==
print("📦 Loading model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=CACHE_DIR,
    device_map=None,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
print(f"✅ Model loaded on: {next(model.parameters()).device}")
model.gradient_checkpointing_enable()
print("🧠 Gradient checkpointing enabled.")

# == Apply LoRA ==
print("⚙️  Applying LoRA config...")
config = LoraConfig(
    r=4,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, config).to(device)
print("✅ LoRA model wrapped.")

# == Dataset ==
print("📝 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"📊 Loaded dataset: {len(dataset)} examples")
dataset = dataset.select(range(min(len(dataset), MAX_EXAMPLES)))
print(f"📉 Trimmed to {len(dataset)} examples for test run")

# == Tokenize ==
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
print("🔎 Sample token:", dataset[0]["input_ids"][:10])

# == Collate function ==
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input": [item["input"] for item in batch],
        "output": [item["output"] for item in batch]
    }

# == Training ==
print("🚦 Starting manual training loop...")
model.train()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

try:
    for step, batch in enumerate(loader):
        print(f"➡️ Step {step+1}/{MAX_STEPS}")
        if step >= MAX_STEPS:
            print("⏹️ Max steps reached. Stopping.")
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        print(f"🔢 Batch shape: {input_ids.shape}")

        print("🧠 Forward pass...")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"📉 Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"✅ Step {step+1} complete\n")

    print("🎉 Training complete.")

except KeyboardInterrupt:
    print("⏹️ Interrupted by user.")
except Exception as e:
    print("❌ Training failed:")
    traceback.print_exc()
    sys.exit(1)

# == Save ==
print(f"💾 Saving to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA saved.")
