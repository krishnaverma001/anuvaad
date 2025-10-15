
import os
import sys
import codecs
import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizerFast

def translate_text(text, model, tokenizer, src_lang="nep_Npi", target_lang="eng_Latn"):
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
    # Construct the absolute path to the model directory to ensure it's found correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "nllb-finetuned-nepali-en")
    
    # --- Model Loading ---
    print("Loading model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = NllbTokenizerFast.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # --- Translation ---
    sentences_to_translate = [
        "मेरो नाम जेमिनी हो।",
        "आज मौसम कस्तो छ?",
        "मलाई नेपाली खाना मन पर्छ।",
        "तपाईंलाई कस्तो छ?",
        "वस्तुको परिवर्तन बचत गर्नुहोस् ।",
        "तिमीलाई कस्तो छ" ,
        "तिमी को हौ",
        "कति बज्यो",
        "बाटो कहाँ छ",
        "फिल्मले सामान्यतया सकारात्मक समीक्षा प्राप्त गर्यो, हिन्दी डब संस्करणमा अत्यन्तै राम्रो प्रदर्शन गर्यो",
        "इङ्गल्याण्डमा भएको गन्तव्य विवाहको पृष्ठभूमिमा सेट गरिएको, कथाले विवाह योजनाकार जगजिन्दर जोगिन्दर र धर्मपुत्र उत्तराधिकारी आलिया अरोरा बीचको विचित्र प्रेमकथालाई पछ्याउँछ, किनकि उनीहरू विचित्र परिवारहरू, व्यक्तिगत आघातहरू र व्यवस्थित विवाहको बेतुकापनहरू पार गर्छन्।",
        "साई रा नरसिंह रेड्डीको वास्तविक कथा रायलसीमा क्षेत्रका एक भारतीय स्वतन्त्रता सेनानी उय्यालवाडा नरसिंह रेड्डीमा केन्द्रित छ जसले १८४६ मा ब्रिटिश इस्ट इन्डिया कम्पनी विरुद्ध पहिलो सामूहिक विद्रोहको नेतृत्व गरेका थिए, सिपाही विद्रोहको एक दशक अघि। एक पोलिगर (एक सामन्ती सरदार), रेड्डी र उनका अनुयायीहरूले कृषि प्रणालीमा शोषणकारी परिवर्तनहरू विरुद्ध विद्रोह गरे, जसमा उनीहरूको पुर्खाको जग्गा कब्जा र कम्पनीद्वारा अनुचित कर लगाउने समावेश थियो। प्रारम्भिक विजय पछि, उनलाई पछि १८४७ मा पक्राउ गरियो र फाँसी दिएर मृत्युदण्ड दिइयो, उनको शरीर डर जग्गाउन प्रदर्शन गरियो।"
    ]

    for sentence in sentences_to_translate:
        print(f"\nOriginal text (Nepali): '{sentence}'")
        translated_text = translate_text(sentence, model, tokenizer)
        print(f"Translated text (English): '{translated_text}'")


if __name__ == "__main__":
    main()
