# train_peft_lora.py

# Setup
import os, sys, traceback, torch, logging

# Ensure parent dir (project root) is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# HuggingFace & PEFT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

# Project config
from config import BASE_MODEL, MAX_TOKENS

# Startindikator
print("🚀 train_peft_lora.py started")

# Loggfix
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Paths och konstanter
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "peft/output_test_lora"
CACHE_DIR = "models/gemma3n"
MAX_EXAMPLES = 2
MAX_STEPS = 1
MAX_TOKENS = 256

print("🧭 Config:")
print(f"  DATA_PATH:     {DATA_PATH}")
print(f"  OUTPUT_DIR:    {OUTPUT_DIR}")
print(f"  CACHE_DIR:     {CACHE_DIR}")
print(f"  MAX_EXAMPLES:  {MAX_EXAMPLES}")
print(f"  MAX_STEPS:     {MAX_STEPS}")

# Initiera enhet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")
if device.type == "cpu":
    print("⚠️  Running on CPU — training will be much slower.")

# Ladda tokenizer och basmodell
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

# Applicera LoRA-konfiguration
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

# Ladda dataset
print("📝 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"📊 Loaded dataset: {len(dataset)} examples")

# Begränsa dataset för testkörning
dataset = dataset.select(range(min(len(dataset), MAX_EXAMPLES)))
print(f"📉 Trimmed to {len(dataset)} examples for test run")

# Tokenisera exempel
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


# Förbered träningsloop
print("🚦 Starting manual training loop...")
model.train()
loader = DataLoader(dataset, batch_size=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Utför träning med tydlig feedback varje steg
try:
    for step, batch in enumerate(loader):
        print(f"➡️ Step {step+1}/{MAX_STEPS}")

        if step >= MAX_STEPS:
            print("⏹️ Max steps reached. Stopping.")
            break

        print(f"📦 Batch keys: {list(batch.keys())}")

        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(device)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).to(device)

        # Kontrollera att shape är [batch_size, seq_len]
        if input_ids.ndim == 1:
            print("⚠️ input_ids is 1D, unsqueezing...")
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            print("⚠️ attention_mask is 1D, unsqueezing...")
            attention_mask = attention_mask.unsqueeze(0)

        print(f"🔢 Input shape: {input_ids.shape} | Attention shape: {attention_mask.shape}")

        labels = input_ids.clone()

        print("🧠 Forward pass...")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

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


# Spara tränad LoRA-adapter
print(f"💾 Saving to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA saved.")
