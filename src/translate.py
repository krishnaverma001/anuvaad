# src/translate.py

# src/translate.py

import torch
from transformers import MBartForConditionalGeneration, NllbTokenizer
import argparse

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Models and Tokenizers ---
print(f"Loading models on {DEVICE.upper()}...")
models = {
    "nepali": MBartForConditionalGeneration.from_pretrained("models/nllb-finetuned-nepali-en").to(DEVICE)
}
tokenizers = {
    "nepali": NllbTokenizer.from_pretrained("models/nllb-finetuned-nepali-en")
}
print("All models loaded successfully!")

def translate_text(text_to_translate: str, source_language: str) -> str:
    """
    Translates a single string of text to English using our fine-tuned models.
    """
    model = models[source_language]
    tokenizer = tokenizers[source_language]

    tokenizer.src_lang = "nep_Npan"

    inputs = tokenizer(text_to_translate, return_tensors="pt").to(DEVICE)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# --- 3. Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a fine-tuned model.")
    parser.add_argument("--text", type=str, required=True, help="Text to translate.")
    parser.add_argument("--lang", type=str, required=True, choices=["nepali"], help="Source language: 'nepali'.")
    args = parser.parse_args()

    translated_sentence = translate_text(args.text, args.lang)
    
    print(f"\nOriginal ({args.lang}): {args.text}")
    print(f"Translated (en): {translated_sentence}")
