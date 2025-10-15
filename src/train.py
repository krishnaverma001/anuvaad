# src/train.py

import os
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

def train_model():
    """
    Fine-tunes a pre-trained NLLB model on a parallel dataset.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a translation model.")
    parser.add_argument("--model_checkpoint", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code (e.g., 'ne')")
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--source_lang_tokenizer", type=str, required=True, help="Source language code for tokenizer (e.g., 'nep_Npan')")
    parser.add_argument("--train_file_source", type=str, required=True, help="Path to the source language training file")
    parser.add_argument("--train_file_target", type=str, required=True, help="Path to the target language training file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    # --- 1. Configuration ---
    MODEL_CHECKPOINT = args.model_checkpoint
    SOURCE_LANG = args.source_lang
    TARGET_LANG = args.target_lang
    MODEL_OUTPUT_DIR = args.output_dir

    # --- 2. Load Tokenizer and Model ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT, src_lang=args.source_lang_tokenizer, tgt_lang="eng_Latn"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # --- 3. Load and Preprocess Data (Memory-Efficiently) ---
    print("Loading and preprocessing data...")
    
    def generate_examples():
        with open(args.train_file_source, "r", encoding="utf-8") as f_src, \
             open(args.train_file_target, "r", encoding="utf-8") as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                yield {"translation": {SOURCE_LANG: src_line.strip(), TARGET_LANG: tgt_line.strip()}}

    dataset = Dataset.from_generator(generate_examples)
    
    split_datasets = dataset.train_test_split(train_size=0.95, seed=42)
    split_datasets["validation"] = split_datasets.pop("test")

    def preprocess_function(examples):
        inputs = [ex[SOURCE_LANG] for ex in examples["translation"]]
        targets = [ex[TARGET_LANG] for ex in examples["translation"]]
        
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=split_datasets["train"].column_names,
    )

    # --- 4. Set Up Training Arguments ---
    print("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False, # Set to True if you have a compatible GPU
    )

    # --- 5. Create the Trainer ---
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. Start Training ---
    print("\n--- Starting model fine-tuning ---")
    trainer.train()
    print("--- Training complete ---")

    # --- 7. Save the Final Model ---
    print(f"Saving final model to {MODEL_OUTPUT_DIR}")
    trainer.save_model()
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
