# scripts/clean_text_data.py

import os
import datetime

def clean_data():
    """
    Reads a raw text file, cleans it, and saves it to the processed data folder.
    """
    # --- Configuration ---
    # Construct the filename based on today's date, matching the scraper's output
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    raw_filename = f"bbc_nepali_articles_{current_date}.txt"
    cleaned_filename = f"bbc_nepali_articles_{current_date}_cleaned.txt"

    # Define the paths using our project structure
    raw_file_path = os.path.join("data", "raw", raw_filename)
    processed_file_path = os.path.join("data", "processed", cleaned_filename)
    
    # Simple rule: we'll discard any line that has fewer than this many words.
    MIN_WORDS_PER_LINE = 5 
    # --- End Configuration ---

    print("--- Starting data cleaning process ---")

    # Check if the raw file exists before we start
    if not os.path.exists(raw_file_path):
        print(f"Error: Raw data file not found at '{raw_file_path}'")
        print("Please run the scraping script first.")
        return

    print(f"Reading raw data from: {raw_file_path}")

    # Read all lines from the raw file
    with open(raw_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # 1. Strip leading/trailing whitespace from the line
        text = line.strip()

        # 2. Apply our cleaning rules
        # We keep the line only if it's not empty AND has enough words
        if text and len(text.split()) >= MIN_WORDS_PER_LINE:
            cleaned_lines.append(text)
    
    # 3. Save the cleaned lines to the new file
    print(f"Saving cleaned data to: {processed_file_path}")
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    with open(processed_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    # Print a summary report
    print("\n--- Cleaning Summary ---")
    print(f"Total lines read: {len(lines)}")
    print(f"Lines after cleaning: {len(cleaned_lines)}")
    print(f"Lines discarded: {len(lines) - len(cleaned_lines)}")
    print("------------------------")

if __name__ == "__main__":
    clean_data()