# scripts/create_test_set.py
import os

# --- Configuration ---
DATA_DIR = "data/processed"
TEST_DIR = "data/test_sets"
SOURCE_FILE = os.path.join(DATA_DIR, "nepali.ne")
TARGET_FILE = os.path.join(DATA_DIR, "nepali.en")
NUM_TEST_LINES = 500
# ---

print("--- Creating a held-back test set for Nepali ---")
os.makedirs(TEST_DIR, exist_ok=True)

# Read all lines from the original files
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    source_lines = f.readlines()
with open(TARGET_FILE, "r", encoding="utf-8") as f:
    target_lines = f.readlines()

# Ensure the files have the same number of lines
assert len(source_lines) == len(target_lines), "Source and target files have different lengths!"

# Split the data
train_source_lines = source_lines[:-NUM_TEST_LINES]
test_source_lines = source_lines[-NUM_TEST_LINES:]

train_target_lines = target_lines[:-NUM_TEST_LINES]
test_target_lines = target_lines[-NUM_TEST_LINES:]

# Write the new, smaller training files (overwriting the old ones)
with open(SOURCE_FILE, "w", encoding="utf-8") as f:
    f.writelines(train_source_lines)
with open(TARGET_FILE, "w", encoding="utf-8") as f:
    f.writelines(train_target_lines)

# Write the new test files
with open(os.path.join(TEST_DIR, "test.ne"), "w", encoding="utf-8") as f:
    f.writelines(test_source_lines)
with open(os.path.join(TEST_DIR, "test.en"), "w", encoding="utf-8") as f:
    f.writelines(test_target_lines)

print(f"Successfully created a test set with {NUM_TEST_LINES} lines for Nepali.")
print(f"The original training files in '{DATA_DIR}' have been updated.")
