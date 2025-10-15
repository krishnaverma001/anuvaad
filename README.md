# Saksi Translation: Nepali-English Machine Translation

This project provides a machine translation solution to translate text from Nepali and Sinhala to English. It leverages the power of the NLLB (No Language Left Behind) model from Meta AI, which is fine-tuned on a custom dataset for improved performance. The project includes a complete workflow from data acquisition to model deployment, featuring a REST API for easy integration.

## Table of Contents

- [Features](#features)
- [Workflow](#workflow)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## Features

-   **High-Quality Translation:** Utilizes a fine-tuned NLLB model for accurate translations.
-   **Support for Multiple Languages:** Currently supports Nepali and Sinhala to English translation.
-   **REST API:** Exposes the translation model through a high-performance FastAPI application.
-   **Interactive Frontend:** A simple and intuitive web interface for easy translation.
-   **Batch Translation:** Supports translating multiple texts in a single request.
-   **Scalable and Reproducible:** Built with a modular structure and uses MLflow for experiment tracking.

## Workflow

The project follows a standard machine learning workflow for building and deploying a translation model:

1.  **Data Acquisition:** The process begins with collecting parallel text data (Nepali/Sinhala and English). The `scripts/fetch_parallel_data.py` script is used to download data from various online sources. The quality and quantity of this data are crucial for the model's performance.

2.  **Data Cleaning and Preprocessing:** Raw data from the web is often noisy and requires cleaning. The `scripts/clean_text_data.py` script performs several preprocessing steps:
    *   **HTML Tag Removal:** Strips out HTML tags and other web artifacts.
    *   **Unicode Normalization:** Normalizes Unicode characters to ensure consistency.
    *   **Sentence Filtering:** Removes sentences that are too long or too short, which can negatively impact training.
    *   **Corpus Alignment:** Ensures a one-to-one correspondence between source and target sentences.

3.  **Model Finetuning:** The core of the project is fine-tuning a pre-trained NLLB model on our custom parallel dataset. The `src/train.py` script, which leverages the Hugging Face `Trainer` API, handles this process. This script manages the entire training loop, including:
    *   Loading the pre-trained NLLB model and tokenizer.
    *   Creating a PyTorch Dataset from the preprocessed data.
    *   Configuring training arguments like learning rate, batch size, and number of epochs.
    *   Executing the training loop and saving the fine-tuned model checkpoints.

4.  **Model Evaluation:** After training, the model's performance is evaluated using the `src/evaluation.py` script. This script calculates the **BLEU (Bilingual Evaluation Understudy)** score, a widely accepted metric for machine translation quality. It works by comparing the model's translations of a test set with a set of high-quality reference translations.

5.  **Inference and Deployment:** Once the model is trained and evaluated, it's ready for use.
    *   `interactive_translate.py`: A command-line script for quick, interactive translation tests.
    *   `fast_api.py`: A production-ready REST API built with FastAPI that serves the translation model. This allows other applications to easily consume the translation service.

## Tech Stack

The technologies used in this project were chosen to create a robust, efficient, and maintainable machine translation pipeline:

-   **Python:** The primary language for the project, offering a rich ecosystem of libraries and frameworks for machine learning.
-   **PyTorch:** A flexible and powerful deep learning framework that provides fine-grained control over the model training process.
-   **Hugging Face Transformers:** The backbone of the project, providing easy access to pre-trained models like NLLB and a standardized interface for training and inference.
-   **Hugging Face Datasets:** Simplifies the process of loading and preprocessing large datasets, with efficient data loading and manipulation capabilities.
-   **FastAPI:** A modern, high-performance web framework for building APIs with Python. It's used to serve the translation model as a REST API.
-   **Uvicorn:** A lightning-fast ASGI server, used to run the FastAPI application.
-   **MLflow:** Used for experiment tracking to ensure reproducibility. It logs training parameters, metrics, and model artifacts, which is crucial for managing machine learning projects.

## Model Details

-   **Base Model:** The project uses the `facebook/nllb-200-distilled-600M` model, a distilled version of the NLLB-200 model. This model is designed to be efficient while still providing high-quality translations for a large number of languages.
-   **Fine-tuning:** The base model is fine-tuned on a custom dataset of Nepali-English and Sinhala-English parallel text to improve its performance on these specific language pairs.
-   **Tokenizer:** The `NllbTokenizer` is used for tokenizing the text. It's a sentence-piece based tokenizer that is specifically designed for the NLLB model.

## API Endpoints

The FastAPI application provides the following endpoints:

-   **`GET /`**: Returns the frontend HTML page.
-   **`GET /languages`**: Returns a list of supported languages.
-   **`POST /translate`**: Translates a single text.
    -   **Request Body:**
        ```json
        {
          "text": "string",
          "source_language": "string"
        }
        ```
    -   **Response Body:**
        ```json
        {
          "original_text": "string",
          "translated_text": "string",
          "source_language": "string"
        }
        ```
-   **`POST /batch-translate`**: Translates a batch of texts.
    -   **Request Body:**
        ```json
        {
          "texts": [
            "string"
          ],
          "source_language": "string"
        }
        ```
    -   **Response Body:**
        ```json
        {
          "original_texts": [
            "string"
          ],
          "translated_texts": [
            "string"
          ],
          "source_language": "string"
        }
        ```

## Getting Started

### Prerequisites

-   **Python 3.10 or higher:** Ensure you have a recent version of Python installed.
-   **Git and Git LFS:** Git is required to clone the repository, and Git LFS is required to handle large model files.
-   **(Optional) NVIDIA GPU with CUDA:** A GPU is highly recommended for training the model.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd saksi_translation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

-   **Fetch Parallel Data:**
    ```bash
    python scripts/fetch_parallel_data.py --output_dir data/raw
    ```

-   **Clean Text Data:**
    ```bash
    python scripts/clean_text_data.py --input_dir data/raw --output_dir data/processed
    ```

### Training

-   **Start Training:**
    ```bash
    python src/train.py \
        --model_name "facebook/nllb-200-distilled-600M" \
        --dataset_path "data/processed" \
        --output_dir "models/nllb-finetuned-nepali-en" \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 8 \
        --num_train_epochs 3
    ```

### Evaluation

-   **Evaluate the Model:**
    ```bash
    python src/evaluate.py \
        --model_path "models/nllb-finetuned-nepali-en" \
        --test_data_path "data/test_sets/test.en" \
        --reference_data_path "data/test_sets/test.ne"
    ```

### Interactive Translation

-   **Run the interactive script:**
    ```bash
    python interactive_translate.py
    ```

### API

-   **Run the API:**
    ```bash
    uvicorn fast_api:app --reload
    ```
    Open your browser and navigate to `http://127.0.0.1:8000` to use the web interface.

## Project Structure

```
saksi_translation/
├── .gitignore
├── fast_api.py             # FastAPI application
├── interactive_translate.py  # Interactive translation script
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── test_translation.py     # Script for testing the translation model
├── frontend/
│   ├── index.html          # Frontend HTML
│   ├── script.js           # Frontend JavaScript
│   └── styles.css          # Frontend CSS
├── data/
│   ├── processed/          # Processed data for training
│   ├── raw/                # Raw data downloaded from the web
│   └── test_sets/          # Test sets for evaluation
├── mlruns/                 # MLflow experiment tracking data
├── models/
│   └── nllb-finetuned-nepali-en/ # Fine-tuned model
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/
│   ├── clean_text_data.py
│   ├── create_test_set.py
│   ├── download_model.py
│   ├── fetch_parallel_data.py
│   └── scrape_bbc_nepali.py
└── src/
    ├── __init__.py
    ├── evaluation.py       # Script for evaluating the model
    ├── train.py            # Script for training the model
    └── translate.py        # Script for translating text
```

## Future Improvements

-   **Support for more languages:** The project can be extended to support more languages by adding more parallel data and fine-tuning the model on it.
-   **Improved Model:** The model can be improved by using a larger version of the NLLB model or by fine-tuning it on a larger and cleaner dataset.
-   **Advanced Frontend:** The frontend can be improved by adding features like translation history, user accounts, and more advanced styling.
-   **Containerization:** The application can be containerized using Docker for easier deployment and scaling.