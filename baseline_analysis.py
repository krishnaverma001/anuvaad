# baseline_analysis.py

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
sinhala_sentences = [
    "ඩෝසන් මිස් දුරකථනයෙන් ඩෝසන් මිස් කවුද සර්",
    "කවුද ඩෝසන් නැතුව ඉන්නේ ඔව් සර්",
    "ඔබ එය උත්සාහ කරන්න සර්",
    "කොහොමද වැඩේ හරිද ඔව් සර්ට ස්තුතියි",
    "ඔව්, හරි, ස්තුතියි රත්තරං"
]

print("\n--- Starting Translation ---")

# Loop through each sentence and translate it.
for sentence in sinhala_sentences:
    
    # 1. Prepare the input for the model
    # We need to tell the tokenizer what the source language is.
    tokenizer.src_lang = "sin_Sinh"
    
    # Convert the text into a format the model understands (input IDs).
    inputs = tokenizer(sentence, return_tensors="pt")

    # 2. Generate the translation
    # We force the model to output English by setting the target language ID.
    target_lang = "eng_Latn"
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.vocab[target_lang],
        max_length=50 # Set a max length for the output
    )

    # 3. Decode the output
    # Convert the model's output tokens back into readable text.
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    # 4. Display the results
    print(f"\nOriginal (si): {sentence}")
    print(f"Translation (en): {translation}")

print("\n--- Translation Complete ---")
