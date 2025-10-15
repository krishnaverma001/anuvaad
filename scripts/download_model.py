
import os
from huggingface_hub import snapshot_download

def download_model():
    """
    Downloads the NLLB model from Hugging Face Hub.
    """
    # --- Configuration ---
    # Note: The original script referred to 'nllb-finetuned-nepali-en', which is not a public model.
    # We are downloading the base model 'facebook/nllb-200-distilled-600M' instead.
    # You may need to fine-tune this model on your own dataset to get the desired performance.
    model_name = "facebook/nllb-200-distilled-600M"
    
    # --- Path setup ---
    # Construct the path to save the model, relative to this script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # We want to save it in saksi_translation/models/nllb-finetuned-nepali-en
    target_dir = os.path.abspath(os.path.join(script_dir, '..', 'models', 'nllb-finetuned-nepali-en'))

    print(f"Downloading model: {model_name}")
    print(f"Saving to: {target_dir}")

    # --- Download ---
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        snapshot_download(repo_id=model_name, local_dir=target_dir, local_dir_use_symlinks=False)
        print("Model downloaded successfully.")
        
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    download_model()
