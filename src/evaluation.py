# src/evaluate.py

import torch
import evaluate # The new, preferred Hugging Face library for metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm # A library to create smart progress bars
import argparse

def evaluate_model():
    """
    Loads a fine-tuned model and evaluates its performance on the test set using the BLEU score.
    """
    parser = argparse.ArgumentParser(description="Evaluate a translation model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--source_lang_file", type=str, required=True, help="Path to the source language test file")
    parser.add_argument("--target_lang_file", type=str, required=True, help="Path to the target language test file")
    parser.add_argument("--source_lang_tokenizer", type=str, required=True, help="Source language code for tokenizer (e.g., 'nep_Npan')")
    args = parser.parse_args()

    # --- 1. Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 2. Load Model, Tokenizer, and Metric ---
    print("Loading model, tokenizer, and evaluation metric...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(DEVICE)
    bleu_metric = evaluate.load("sacrebleu")

    # --- 3. Load Test Data ---
    with open(args.source_lang_file, "r", encoding="utf-8") as f:
        source_sentences = [line.strip() for line in f.readlines()]
    with open(args.target_lang_file, "r", encoding="utf-8") as f:
        # The BLEU metric expects references to be a list of lists
        reference_translations = [[line.strip()] for line in f.readlines()]

    # --- 4. Generate Predictions ---
    print(f"Generating translations for {len(source_sentences)} test sentences...")
    predictions = []
    for sentence in tqdm(source_sentences):
        tokenizer.src_lang = args.source_lang_tokenizer
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.vocab["eng_Latn"],
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