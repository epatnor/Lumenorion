# train_peft_lora.py

# == Imports ==
import os, sys, traceback, torch, logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import BASE_MODEL, MAX_TOKENS


# == Paths and constants ==
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "lora_training/outputs/gemma3n_lora_test"
CACHE_DIR = "models/gemma3n"
MAX_EXAMPLES = 40
MAX_STEPS = 20
BATCH_SIZE = 2


# == Init ==
print("🔬 Training LoRA model...")
print("🚀 train_peft_lora.py started")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Show config
print("🧭 Config:")
print(f"  DATA_PATH:     {DATA_PATH}")
print(f"  OUTPUT_DIR:    {OUTPUT_DIR}")
print(f"  CACHE_DIR:     {CACHE_DIR}")
print(f"  MAX_EXAMPLES:  {MAX_EXAMPLES}")
print(f"  MAX_STEPS:     {MAX_STEPS}")
print(f"  BATCH_SIZE:    {BATCH_SIZE}")

# Init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")
if device.type == "cpu":
    print("⚠️  Running on CPU — training will be much slower.")

# == Load tokenizer & model ==
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

# Enable memory saving
if hasattr(model, "gradient_checkpointing_enable"):
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

# == Load and prepare dataset ==
print("📝 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"📊 Loaded dataset: {len(dataset)} examples")
dataset = dataset.select(range(min(len(dataset), MAX_EXAMPLES)))
print(f"📉 Trimmed to {len(dataset)} examples for test run")

def tokenize(batch):
    texts = []
    for input_text, output_text in zip(batch["input"], batch["output"]):
        input_text = " ".join(input_text) if isinstance(input_text, list) else input_text
        output_text = " ".join(output_text) if isinstance(output_text, list) else output_text
        texts.append(f"{input_text}\n{output_text}")
    return tokenizer(texts, truncation=True, max_length=MAX_TOKENS, padding="max_length")

print("✍️  Tokenizing...")
dataset = dataset.map(tokenize, batched=True, num_proc=1)
print("✅ Tokenization complete.")
print("🔎 Sample token:", dataset[0]["input_ids"][:10])

# == Training ==
print("🚦 Starting manual training loop...")
model.train()
loader = DataLoader(dataset, batch_size=BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

interrupted = False

try:
    for step, batch in enumerate(loader):
        if step >= MAX_STEPS:
            print("⏹️ Max steps reached. Stopping.")
            break

        print(f"➡️ Step {step+1}/{MAX_STEPS}")

        # Hantera batch oavsett typ (list eller tensor)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if isinstance(input_ids, list):
            print("⚠️ input_ids is list, converting to tensor...")
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if isinstance(attention_mask, list):
            print("⚠️ attention_mask is list, converting to tensor...")
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        print(f"🔢 Batch shape: {input_ids.shape}")
        print("🧠 Forward pass...")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone()
        )

        loss = outputs.loss
        print(f"📉 Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"✅ Step {step+1} complete\n")

except KeyboardInterrupt:
    interrupted = True
    print("⏹️ Training manually interrupted by user.")
except Exception:
    print("❌ Training failed:")
    traceback.print_exc()
    sys.exit(1)

# == Post-training ==
if interrupted:
    print("⚠️ Training was interrupted before completion.")
else:
    print("🎉 Training complete.")

# Save trained LoRA
print(f"💾 Saving to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA saved.")
