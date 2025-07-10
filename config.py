# config.py

# Base model for inference and training
BASE_MODEL = "google/gemma-3n-E2B-it"

# Local cache override (optional)
USE_LOCAL_MODEL = False
LOCAL_MODEL_PATH = "path/to/your/ollama_export"

# Token limits, other shared constants
MAX_TOKENS = 32000
TEMPERATURE = 0.7
TOP_P = 0.9
