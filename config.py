# config.py

# Base model for inference and training
BASE_MODEL = "google/gemma-3n-E2B-it"

# Use local model instead of downloading from Hugging Face
USE_LOCAL_MODEL = False
LOCAL_MODEL_PATH = "path/to/your/ollama_export"

# Sampling and generation parameters
MAX_TOKENS = 1024          # 👈 Kraftigt sänkt från 32000 för att minska GPU-minne
TEMPERATURE = 0.7
TOP_P = 0.9

# LoRA fine-tuning output path
LORA_DIR = "./output_gemma_lora"

# Dataset paths
DATA_PATH = "lora_training/datasets/lumenorion_lora.jsonl"
SHUFFLED_DATA_PATH = "lora_training/datasets/lumenorion_lora_shuffled.jsonl"
STATS_PATH = "lora_training/datasets/stats.json"

# Database and file logging
DB_PATH = "lumenorion.db"
DREAM_DIR = "lora_training/dreams"
REFLECT_DIR = "lora_training/reflections"
REFLECT_LOG_DIR = "lora_training/logs_reflect"
CONVO_DIR = "lora_training/conversations"
STATE_PATH = "state.json"

# Training hyperparameters
EPOCHS = 1             # 👈 Endast 1 epoch vid testkörning, kan höjas sen
BATCH_SIZE = 1         # ✅ Redan optimalt för låg VRAM
LEARNING_RATE = 1e-4   # 👈 Lägre learning rate för mindre volatila FP16-beräkningar
MAX_SEQ_LENGTH = 512   # 👈 Vid ännu tightare VRAM – matcha med MAX_TOKENS
FP16 = True            # ✅ Behåll
