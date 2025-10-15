# src/evaluate_sinhala.py

import torch
import evaluate # The new, preferred Hugging Face library for metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm # A library to create smart progress bars

def evaluate_model():
    """
    Loads a fine-tuned model and evaluates its performance on the test set using the BLEU score.
    """
    # --- 1. Configuration ---
    MODEL_PATH = "thilina/mt5-sinhalese-english"
    TEST_DIR = "data/test_sets"
    SOURCE_LANG_FILE = f"{TEST_DIR}/test.si"
    TARGET_LANG_FILE = f"{TEST_DIR}/test.en"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 2. Load Model, Tokenizer, and Metric ---
    print("Loading model, tokenizer, and evaluation metric...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    bleu_metric = evaluate.load("sacrebleu")

    # --- 3. Load Test Data ---
    with open(SOURCE_LANG_FILE, "r", encoding="utf-8") as f:
        source_sentences = [line.strip() for line in f.readlines()]
    with open(TARGET_LANG_FILE, "r", encoding="utf-8") as f:
        # The BLEU metric expects references to be a list of lists
        reference_translations = [[line.strip()] for line in f.readlines()]

    # --- 4. Generate Predictions ---
    print(f"Generating translations for {len(source_sentences)} test sentences...")
    predictions = []
    for sentence in tqdm(source_sentences):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        
        generated_tokens = model.generate(
            **inputs,
            max_length=128
        )
        
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        predictions.append(translation)

    # --- 5. Compute BLEU Score ---
    print("Calculating BLEU score...")
    results = bleu_metric.compute(predictions=predictions, references=reference_translations)
    
    # The result is a dictionary. The 'score' key holds the main BLEU score.
    bleu_score = results["score"]

    print("\n--- Evaluation Complete ---")
    print(f"BLEU Score: {bleu_score:.2f}")
    print("---------------------------")

if __name__ == "__main__":
    evaluate_model()
