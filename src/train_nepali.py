# src/train_nepali.py

import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

def train_nepali_model():
    """
    Fine-tunes a pre-trained NLLB model on the Nepali parallel dataset.
    """
    # --- 1. Configuration ---
    MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"
    DATA_DIR = "data/processed"
    MODEL_OUTPUT_DIR = "D:\\SIH\\models\\nllb-finetuned-nepali-en"

    # --- 2. Load Tokenizer and Model ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT, src_lang="nep_Npan", tgt_lang="eng_Latn"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # --- 3. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    nepali_dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "nepali.ne"))["train"]
    english_dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "nepali.en"))["train"]

    # rename the 'text' column to 'ne' and 'en'
    nepali_dataset = nepali_dataset.rename_column("text", "ne")
    english_dataset = english_dataset.rename_column("text", "en")

    # combine the datasets
    raw_datasets = concatenate_datasets([nepali_dataset, english_dataset], axis=1)
    
    split_datasets = raw_datasets.train_test_split(train_size=0.95, seed=42)
    split_datasets["validation"] = split_datasets.pop("test")

    def preprocess_function(examples):
        inputs = examples["ne"]
        targets = examples["en"]
        
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3, # Reduced for faster training, can be increased
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
    print(f"\n--- Starting model fine-tuning for Nepali-English ---")
    trainer.train()
    print("--- Training complete ---")

    # --- 7. Save the Final Model ---
    print(f"Saving final model to {MODEL_OUTPUT_DIR}")
    trainer.save_model()
    print("Model saved successfully!")

if __name__ == "__main__":
    train_nepali_model()