"""
An interactive script to translate text to English using a fine-tuned NLLB model.
"""

import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizer

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_LANGUAGES = {
    "nepali": "nep_Npan",
    "sinhala": "sin_Sinh",
}

# --- 2. Load Model and Tokenizer ---
def load_model_and_tokenizer(model_path):
    """Loads the model and tokenizer from the given path."""
    print(f"Loading model on {DEVICE.upper()}...")
    try:
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        tokenizer = NllbTokenizer.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# --- 3. Translation Function ---
def translate_text(model, tokenizer, text: str, src_lang: str) -> str:
    """
    Translates a single string of text to English.
    """
    if src_lang not in SUPPORTED_LANGUAGES:
        return f"Language '{src_lang}' not supported."

    tokenizer.src_lang = SUPPORTED_LANGUAGES[src_lang]
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128,
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# --- 4. Interactive Translation Loop ---
if __name__ == "__main__":
    # Select model path based on language
    lang_choice = input(f"Choose a language ({list(SUPPORTED_LANGUAGES.keys())}): ").lower()
    if lang_choice not in SUPPORTED_LANGUAGES:
        print("Invalid language choice.")
        exit()

    # For now, we assume a single model path. This can be extended.
    model_path = "models/nllb-finetuned-nepali-en"
    model, tokenizer = load_model_and_tokenizer(model_path)

    if model and tokenizer:
        print(f"\n--- Interactive Translation ({lang_choice.capitalize()}) ---")
        print(f"Enter a {lang_choice} sentence to translate to English.")
        print("Type 'exit' to quit.\n")

        while True:
            text_to_translate = input(f"{lang_choice.capitalize()}: ")
            if text_to_translate.lower() == "exit":
                break

            if not text_to_translate.strip():
                print("Please enter some text to translate.")
                continue

            english_translation = translate_text(model, tokenizer, text_to_translate, lang_choice)
            print(f"English: {english_translation}\n")