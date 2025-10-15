# scripts/fetch_parallel_data.py

from datasets import load_dataset
import os


def fetch_and_save_parallel_data(lang_pair, dataset_name, output_name):
    """
    Downloads a parallel dataset and saves it into two
    separate text files (one for each language).
    
    Args:
        lang_pair (str): Language pair, e.g., "en-ne" for English-Nepali.
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        output_name (str): The name to use for the output files.
    """
    source_lang, target_lang = lang_pair.split("-")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    source_filepath = os.path.join(output_dir, f"{output_name}.{source_lang}")
    target_filepath = os.path.join(output_dir, f"{output_name}.{target_lang}")

    print(f"--- Starting download for {lang_pair} from {dataset_name} ---")
    
    try:
        # Load the dataset from Hugging Face
        if dataset_name == "Programmer-RD-AI/sinhala-english-singlish-translation":
            dataset = load_dataset(dataset_name, split='train')
        else:
            dataset = load_dataset(dataset_name, lang_pair, split='train')
        print(f"Dataset loaded successfully. Total pairs: {len(dataset)}")

        print(f"Processing and saving files...")
        with open(source_filepath, "w", encoding="utf-8") as f_source, \
             open(target_filepath, "w", encoding="utf-8") as f_target:
            
            for example in dataset:
                if dataset_name == "Programmer-RD-AI/sinhala-english-singlish-translation":
                    source_sentence = example['Sinhala']
                    target_sentence = example['English']
                else:
                    source_sentence = example['translation'][source_lang]
                    target_sentence = example['translation'][target_lang]

                if source_sentence and target_sentence:
                    f_source.write(source_sentence.strip() + "\n")
                    f_target.write(target_sentence.strip() + "\n")

        print(f"Successfully saved data for {lang_pair}")
    except Exception as e:
        print(f"An error occurred for {lang_pair}: {e}")

if __name__ == "__main__":
    # --- Fetch Nepali Data ---
    print("Fetching Nepali data...")
    fetch_and_save_parallel_data(lang_pair="en-ne", dataset_name="Helsinki-NLP/opus-100", output_name="nepali")
    
    # --- Fetch Sinhalese Data ---
    print("\nFetching Sinhalese data...")
    fetch_and_save_parallel_data(lang_pair="si-en", dataset_name="Programmer-RD-AI/sinhala-english-singlish-translation", output_name="sinhala")
    

    # --- Fetch Sinhalese Idioms Data ---
    print("\nFetching Sinhalese idioms data...")
    output_dir = "data/processed"
    try:
        idioms_dataset = load_dataset("Venuraa/English-Sinhala-Idioms-Parallel-Translations", split='train')
        print(f"Idioms dataset loaded successfully. Total pairs: {len(idioms_dataset)}")

        with open(os.path.join(output_dir, "sinhala.si"), "a", encoding="utf-8") as f_source, \
             open(os.path.join(output_dir, "sinhala.en"), "a", encoding="utf-8") as f_target:
            for example in idioms_dataset:
                parts = example['text'].split('\n')
                if len(parts) == 2:
                    f_target.write(parts[0] + "\n")
                    f_source.write(parts[1] + "\n")
        print("Successfully appended idioms data.")
    except Exception as e:
        print(f"An error occurred while fetching idioms data: {e}")
    print("\nAll data fetching complete.")
