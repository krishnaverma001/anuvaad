"""
A FastAPI application for serving the translation model, inspired by interactive_translate.py.
"""

import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from typing import List
import fitz  # PyMuPDF
import shutil
import os

# --- 1. App Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Saksi Translation API",
    description="A simple API for translating text and PDFs to English.",
    version="2.0",
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# --- 2. Global Variables ---

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

SUPPORTED_LANGUAGES = {
    "nepali": "nep_Npan",
    "sinhala": "sin_Sinh",
}

MODEL_PATH = "krishnaverma01/anuvaad"

model = None
tokenizer = None

# --- 3. Pydantic Models ---
class TranslationRequest(BaseModel):
    text: str
    source_language: str

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_language: str

class BatchTranslationResponse(BaseModel):
    original_texts: List[str]
    translated_texts: List[str]
    source_language: str
    
class PdfTranslationResponse(BaseModel):
    filename: str
    translated_text: str
    source_language: str


# --- 4. Helper Functions ---
def load_model_and_tokenizer(model_path):
    """Loads the model and tokenizer from the given path."""
    global model, tokenizer
    logger.info(f"Loading model on {DEVICE.upper()}...")
    try:
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        tokenizer = NllbTokenizer.from_pretrained(model_path)
        logger.info("Model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # In a real app, you might want to exit or handle this more gracefully
        raise

def translate_text(text: str, src_lang: str) -> str:
    """
    Translates a single string of text to English.
    """
    if src_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language '{src_lang}' not supported.")

    tokenizer.src_lang = SUPPORTED_LANGUAGES[src_lang]
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128,
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def batch_translate_text(texts: List[str], src_lang: str) -> List[str]:
    """
    Translates a batch of texts to English.
    """
    if src_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language '{src_lang}' not supported.")

    tokenizer.src_lang = SUPPORTED_LANGUAGES[src_lang]
    # We use padding=True to handle batches of different lengths
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=512, # Allow for longer generated sequences in batches
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# --- 5. API Events ---
@app.on_event("startup")
async def startup_event():
    """Load the model at startup."""
    load_model_and_tokenizer(MODEL_PATH)

# --- 6. API Endpoints ---
@app.get("/")
async def root():
    """Returns the frontend."""
    return FileResponse('frontend/index.html')

@app.get("/languages")
def get_supported_languages():
    """Returns a list of supported languages."""
    return {"supported_languages": list(SUPPORTED_LANGUAGES.keys())}

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translates a single text from a source language to English."""
    try:
        translated_text = translate_text(request.text, request.source_language)
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate(request: BatchTranslationRequest):
    """Translates a batch of texts from a source language to English."""
    try:
        translated_texts = batch_translate_text(request.texts, request.source_language)
        return BatchTranslationResponse(
            original_texts=request.texts,
            translated_texts=translated_texts,
            source_language=request.source_language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/translate-pdf", response_model=PdfTranslationResponse)
async def translate_pdf(source_language: str, file: UploadFile = File(...)):
    """Translates a PDF file from a source language to English."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    # Save the uploaded file temporarily
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract text from the PDF
        doc = fitz.open(temp_pdf_path)
        extracted_text = ""
        for page in doc:
            extracted_text += page.get_text()
        doc.close()

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")

        # Split text into chunks (e.g., by paragraph) to handle large texts
        text_chunks = [p.strip() for p in extracted_text.split('\n') if p.strip()]
        
        # Translate the chunks in batches
        translated_chunks = batch_translate_text(text_chunks, source_language)

        # Join the translated chunks back together
        final_translation = "\n".join(translated_chunks)

        return PdfTranslationResponse(
            filename=file.filename,
            translated_text=final_translation,
            source_language=source_language,
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


# --- 7. Example Usage (for running with uvicorn) ---
# To run this API, use the following command in your terminal:
# uvicorn fast_api:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
