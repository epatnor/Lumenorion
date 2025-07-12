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

# Loggfix
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Paths och konstanter
DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
OUTPUT_DIR = "peft/output_gemma_lora"
CACHE_DIR = "models/gemma3n"
MAX_EXAMPLES = 20
MAX_STEPS = 50

# Initiera enhet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

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

# Förbered träningsloop
print("🚦 Starting manual training loop...")
model.train()
loader = DataLoader(dataset, batch_size=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Utför träning med tydlig feedback varje steg
try:
    for step, batch in enumerate(loader):
        if step >= MAX_STEPS:
            print("⏹️ Max steps reached. Stopping.")
            break

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"🔁 Step {step+1}/{MAX_STEPS} | Loss: {loss.item():.4f}")

    print("✅ Training complete.")

except KeyboardInterrupt:
    print("⏹️ Interrupted by user.")
except Exception as e:
    print("❌ Training failed:")
    traceback.print_exc()
    sys.exit(1)

# Spara tränad LoRA-adapter
print(f"💾 Saving to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA saved.")
