# debug_load.py

import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
nepali_model_path = r"D:\SIH\saksi_translation\models\nllb-finetuned-nepali-en"

# --- Tokenizer Loading ---
print("Loading Nepali tokenizer...")
try:
    nepali_tokenizer = AutoTokenizer.from_pretrained(nepali_model_path)
    print("Nepali tokenizer loaded successfully.")
    print(nepali_tokenizer)
except Exception as e:
    print(f"Error loading Nepali tokenizer: {e}")

# --- Model Loading ---
print("\nLoading Nepali model...")
try:
    nepali_model = M2M100ForConditionalGeneration.from_pretrained(nepali_model_path).to(DEVICE)
    print("Nepali model loaded successfully.")
    print(nepali_model)
except Exception as e:
    print(f"Error loading Nepali model: {e}")
