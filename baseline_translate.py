# baseline_translate.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Define the model we want to use. We'll use a distilled (smaller, faster)
# version of NLLB-200 for this quick test.
model_name = "facebook/nllb-200-distilled-600M"

# Load the pre-trained tokenizer and model from Hugging Face.
# This might take a minute to download the first time.
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model loaded successfully!")

# Sentences we want to translate.
sentences_to_translate = {
    "nep_Npan": "नेपालको राजधानी काठमाडौं हो।",  # Nepali: "The capital of Nepal is Kathmandu."
    "sin_Sinh": "ශ්‍රී ලංකාවේ අගනුවර කොළඹ වේ."   # Sinhala: "The capital of Sri Lanka is Colombo."
}

print("\n--- Starting Translation ---")

# Loop through each sentence and translate it.
for lang_code, text in sentences_to_translate.items():
    
    # 1. Prepare the input for the model
    # We need to tell the tokenizer what the source language is.
    tokenizer.src_lang = lang_code
    
    # Convert the text into a format the model understands (input IDs).
    inputs = tokenizer(text, return_tensors="pt")

    # 2. Generate the translation
    # We force the model to output English by setting the target language ID.
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=50 # Set a max length for the output
    )

    # 3. Decode the output
    # Convert the model's output tokens back into readable text.
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    # 4. Display the results
    print(f"\nOriginal ({lang_code}): {text}")
    print(f"Translation (eng_Latn): {translation}")

print("\n--- Translation Complete ---")