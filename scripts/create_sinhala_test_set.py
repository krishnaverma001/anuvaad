# scripts/create_sinhala_test_set.py
import os
from datasets import load_dataset

# --- Configuration ---
DATA_DIR = "data/processed"
TEST_DIR = "data/test_sets"
DATASET_NAME = "Programmer-RD-AI/sinhala-english-singlish-translation"
NUM_TEST_LINES = 500
# ---

print("--- Creating a held-back test set for Sinhalese ---")
os.makedirs(TEST_DIR, exist_ok=True)

# Load the dataset from Hugging Face
dataset = load_dataset(DATASET_NAME, split='train')

# Split the dataset
train_dataset = dataset.select(range(len(dataset) - NUM_TEST_LINES))
test_dataset = dataset.select(range(len(dataset) - NUM_TEST_LINES, len(dataset)))

# Write the new training files
with open(os.path.join(DATA_DIR, "sinhala.si"), "w", encoding="utf-8") as f_source, \
     open(os.path.join(DATA_DIR, "sinhala.en"), "w", encoding="utf-8") as f_target:
    for example in train_dataset:
        f_source.write(example['Sinhala'] + "\n")
        f_target.write(example['English'] + "\n")

# Write the new test files
with open(os.path.join(TEST_DIR, "test.si"), "w", encoding="utf-8") as f_source, \
     open(os.path.join(TEST_DIR, "test.en"), "w", encoding="utf-8") as f_target:
    for example in test_dataset:
        f_source.write(example['Sinhala'] + "\n")
        f_target.write(example['English'] + "\n")

print(f"Successfully created a test set with {NUM_TEST_LINES} lines for Sinhalese.")
print(f"The original training files in '{DATA_DIR}' have been updated.")
