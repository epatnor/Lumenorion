# train_peft_lora.py

# == Imports ==
import os, sys, traceback, torch, logging, time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import BASE_MODEL, MAX_TOKENS, LORA_DIR, DATA_PATH

# == Init ==
print("🔬 Training LoRA model...")
print("🚀 train_peft_lora.py started")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# == Paths and constants ==
OUTPUT_DIR = LORA_DIR
CACHE_DIR = "models/gemma3n"
MAX_EXAMPLES = 40
MAX_STEPS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

print("🧭 Config:")
print(f"  DATA_PATH:     {DATA_PATH}")
print(f"  OUTPUT_DIR:    {OUTPUT_DIR}")
print(f"  CACHE_DIR:     {CACHE_DIR}")
print(f"  MAX_EXAMPLES:  {MAX_EXAMPLES}")
print(f"  MAX_STEPS:     {MAX_STEPS}")
print(f"  BATCH_SIZE:    {BATCH_SIZE}")

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
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

print("✍️  Tokenizing...")
dataset = dataset.map(tokenize, batched=True, num_proc=1)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
print("✅ Tokenization complete.")
print("🔎 Sample token:", dataset[0]["input_ids"][:10])

# == Training ==
print("🚦 Starting manual training loop...")
model.train()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=default_data_collator)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

steps_to_run = min(MAX_STEPS, len(loader))
if steps_to_run < MAX_STEPS:
    print(f"ℹ️ Adjusted MAX_STEPS to {steps_to_run} (based on dataset size)")

interrupted = False

try:
    for step, batch in enumerate(loader):
        if step >= steps_to_run:
            print("⏹️ Max steps reached. Stopping.")
            break

        print(f"➡️ Step {step+1}/{steps_to_run}", flush=True)
        step_start = time.time()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        print(f"🔢 Batch shape: {input_ids.shape}", flush=True)
        print("🧠 Forward pass...", flush=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone()
        )

        loss = outputs.loss
        print(f"📉 Loss: {loss.item():.4f}", flush=True)
        print("🔁 Backward pass...", flush=True)
        loss.backward()

        print("🔧 Optimizer step...", flush=True)
        optimizer.step()

        print("🧹 Zeroing gradients...", flush=True)
        optimizer.zero_grad()

        step_end = time.time()
        print(f"✅ Step {step+1} complete | 🕒 {step_end - step_start:.2f}s\n", flush=True)

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

print(f"💾 Saving to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA saved.")

# == Quick evaluation ==
print("\n🔍 Running eval preview...")

try:
    model.eval()
    eval_sample = dataset[-1]
    prompt = tokenizer.decode(eval_sample["input_ids"], skip_special_tokens=True)
    expected_output = eval_sample.get("output", "[No expected output in sample]")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n🧪 Prompt:")
    print(prompt)
    print("\n🎯 Expected:")
    print(expected_output)
    print("\n🤖 Model output:")
    print(generated_text)

except Exception as e:
    print("❌ Eval preview failed:")
    print(e)
