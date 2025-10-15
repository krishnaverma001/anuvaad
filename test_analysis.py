import os
import sys
import codecs
import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizerFast

def translate_text(text, model, tokenizer, src_lang, target_lang="eng_Latn"):
    """
    Translates a single text string.
    """
    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.vocab[target_lang],
            max_length=512
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        return f"An error occurred during translation: {e}"

def main():
    """
    Main function to load the model and run a test translation.
    """
    # Reconfigure stdout to handle UTF-8 encoding
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nepali_model_path = os.path.join(script_dir, "models", "nllb-finetuned-nepali-en")
    
    # --- Model Loading ---
    print("Loading Nepali model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nepali_model = M2M100ForConditionalGeneration.from_pretrained(nepali_model_path).to(device)
        nepali_tokenizer = NllbTokenizerFast.from_pretrained(nepali_model_path)
        print("Nepali model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading Nepali model or tokenizer: {e}")
        return

    # --- Nepali Translation ---
    nepali_sentences = [
        "जडान बिन्दु थप्नुहोस्",
        "स्टिकी नोट आयात पूरा भयो",
        "मोनोस्पेस १२",
        "पानी जेट पम्पमा दुईवटा भित्रिने र एउटा बाहिरिने पाइप हुन्छन् र एक भित्र अर्को सिद्धान्त अनुरूप दुईवटा पाइप हुन्छन् । पानीको प्रविष्टिमा एउटा पानी जेटले केही ठूलो पाइपमा पूरा चापले टुटीबाट बाहिर फाल्दछ । यस्तो तरिकाले पानी जेटले वायू वा तरललाई दोस्रो प्रविष्टिबाट टाढा पुर्याउदछ । ड्रिफ्टिङ तरलमा ऋणात्मक चापको कारणले यस्तो हुन्छ । त्यसैले यो हाइड्रोडायनमिक विरोधाभाषको एउटा अनुप्रयोग हो । यसले ड्रिफ्टिङ तरल नजिकका वस्तु टाढा फाल्नुको साटोमा सोस्ने कुरा बताउदछ ।",
        "वस्तुको परिवर्तन बचत गर्नुहोस् ।"
        "तिमीलाई कस्तो छ" ,
        "तिमी को हौ",
        "कति बज्यो"
    ]

    print("\n--- Nepali to English Translation Analysis ---")
    for sentence in nepali_sentences:
        print(f"\nOriginal (ne): {sentence}")
        translated_text = translate_text(sentence, nepali_model, nepali_tokenizer, src_lang="nep_Npan")
        print(f"Translated (en): {translated_text}")

    # --- Sinhala Translation ---
    # NOTE: No fine-tuned model for sinhala was found. Using the baseline model for now.
    print("\n\n--- Sinhala to English Translation Analysis ---")
    
    sinhala_sentences = [
        "ඩෝසන්මිස් දුරකථනයෙන් ඩෝසන්මිස් කවුද සර්",
        "කවුද ඩෝසන් නැතුව ඉන්නේ ඔව් සර්",
        "ඔබ එය උත්සාහ කරන්න සර්",
        "කොහොමද වැඩේ හරිද ඔව් සර්ට ස්තුතියි",
        "ඔව්, හරි, ස්තුතියි රත්තරං",

    ]

    for sentence in sinhala_sentences:
        print(f"\nOriginal (si): {sentence}")
        translated_text = translate_text(sentence, nepali_model, nepali_tokenizer, src_lang="sin_Sinh")
        print(f"Translated (en): {translated_text}")


if __name__ == "__main__":
    main()
