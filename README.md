# English → Malayalam Neural Machine Translation

A local English-to-Malayalam neural machine translation application built using the NLLB-200 1.3B model, CTranslate2 INT8 inference, FastAPI, and a lightweight web interface.

The application provides a simple browser-based interface for translating English text into Malayalam locally.

---

## Overview

This project provides a local inference application for English → Malayalam neural machine translation.

The translation pipeline is:

```text
English Text
     ↓
NLLB Tokenizer
     ↓
CTranslate2 INT8 Model
     ↓
Malayalam Text
```

### Supported Translation Direction

```text
English (eng_Latn) → Malayalam (mal_Mlym)
```

The current application supports **English → Malayalam only**.

---

## Features

- English → Malayalam neural machine translation
- NLLB-200 1.3B model
- CTranslate2 INT8 inference
- FastAPI backend
- Browser-based translation interface
- Adjustable beam search
- Maximum output token configuration
- Translation API
- Batch translation API
- FastAPI Swagger documentation
- Health-check endpoint
- Local inference without an external translation API
- CPU-based inference
- Responsive light/dark web interface

---

## Architecture

```text
                    ┌─────────────────────┐
                    │    Web Interface    │
                    │   HTML / CSS / JS   │
                    └──────────┬──────────┘
                               │
                               │ HTTP
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   NLLB Tokenizer    │
                    │     eng_Latn        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    CTranslate2      │
                    │      INT8 Model     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Malayalam Output    │
                    │      mal_Mlym       │
                    └─────────────────────┘
```

---

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- CTranslate2
- Hugging Face Transformers
- SentencePiece

### Frontend

- HTML
- CSS
- JavaScript

### Model

- NLLB-200 Distilled 1.3B
- CTranslate2 INT8

---

## Model

The application uses the following CTranslate2 model:

```text
Navneeth017/nllb-1.3b-en-ml-ct2
```

The model is based on:

```text
facebook/nllb-200-distilled-1.3B
```

### Model Configuration

```text
Source language : eng_Latn
Target language : mal_Mlym
Model           : NLLB-200 Distilled 1.3B
Inference       : CTranslate2
Quantization    : INT8
Current device  : CPU
```

According to the model documentation, the model was fine-tuned on English-Malayalam data and reports the following held-out validation results:

```text
BLEU    : 33.20
chrF++  : 62.62
```

These are the model author's reported in-domain evaluation results.

---

## Example Translation

### Input

```text
Heavy rain is expected in all districts of Kerala tomorrow.
```

### Output

```text
കേരളത്തിലെ എല്ലാ ജില്ലകളിലും നാളെ കനത്ത മഴ പ്രതീക്ഷിക്കുന്നു.
```

---

## Project Structure

```text
malayalam-nmt/
│
├── app.py
├── requirements.txt
├── run.sh
├── install.sh
├── .env.example
├── .gitignore
├── README.md
│
└── static/
    ├── index.html
    ├── app.js
    └── styles.css
```

---

## Requirements

- Linux / WSL2
- Python 3.12+
- pip
- Git
- Sufficient RAM for loading the model

The current configuration uses CPU-based CTranslate2 INT8 inference.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nandanap24/malayalam-nmt.git
cd malayalam-nmt
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv ~/nllb_fastapi_env
source ~/nllb_fastapi_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Setup

The model is **not included in this Git repository** because of its large size.

Download or obtain the CTranslate2 model separately and place it in a local directory.

The model directory should contain files similar to:

```text
nllb-1.3b-en-ml-ct2/
├── config.json
├── model.bin
├── shared_vocabulary.json
├── tokenizer.json
├── tokenizer_config.json
├── sentencepiece.bpe.model
└── ...
```

Configure the model location using the environment configuration.

---

## Configuration

Create a `.env` file based on `.env.example`.

Example:

```env
MODEL_DIR=/path/to/nllb-1.3b-en-ml-ct2
CT2_DEVICE=cpu
CT2_COMPUTE_TYPE=int8
```

| Variable | Description | Example |
|---|---|---|
| `MODEL_DIR` | Path to the CTranslate2 model | `/path/to/model` |
| `CT2_DEVICE` | CTranslate2 execution device | `cpu` |
| `CT2_COMPUTE_TYPE` | Computation type | `int8` |

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or use:

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8000
```

---

## API Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /api/health
```

Returns the current backend and model configuration.

### Translate Text

```http
POST /api/translate
```

Example request:

```json
{
  "text": "Artificial intelligence can improve language technology.",
  "beam_size": 4,
  "max_new_tokens": 256
}
```

### Batch Translation

```http
POST /api/translate-batch
```

Supports translating multiple English inputs in a single request.

---

## Beam Search

The application supports configurable beam search during generation.

A larger beam size can improve search quality in some cases, but it can also increase inference time.

Example:

```json
{
  "beam_size": 4
}
```

---

## Performance

The current configuration uses:

```text
Device       : CPU
Compute type : INT8
Inference    : CTranslate2
```

Inference time depends on input length, output length, beam size, CPU hardware, system memory, and CTranslate2 configuration.

The application is intended primarily as a **local inference and demonstration application**, rather than a high-throughput production translation service.

---

## Repository Design

Large model files are intentionally excluded from Git.

The `.gitignore` excludes model formats such as:

```text
*.bin
*.safetensors
*.onnx
```

It also excludes local model directories and environment files.

This keeps the Git repository lightweight while allowing the model to be stored separately.

---

## Limitations

- Current translation direction is English → Malayalam only.
- CPU inference can take several seconds depending on input length and beam size.
- Translation quality may vary for sentences outside the model's training and evaluation domain.
- The model is not included directly in this Git repository because of its size.
- The current application is designed primarily for local inference.

---

## Model Attribution

This application uses:

```text
Navneeth017/nllb-1.3b-en-ml-ct2
```

The model is based on:

```text
facebook/nllb-200-distilled-1.3B
```

Please refer to the respective model repositories for complete model, dataset, and licensing information.

The provided `Navneeth017/nllb-1.3b-en-ml-ct2` model documentation specifies **CC BY-NC 4.0** licensing.

---

## Future Improvements

- GPU inference support
- Translation history
- Improved batch processing
- Automated translation quality evaluation
- Further model optimization
- Production deployment
- Authentication and usage monitoring
- Additional language pairs

---

## Author

**Nandana P**

GitHub: https://github.com/nandanap24

---

## License

The application code in this repository is provided for educational and portfolio purposes.

The underlying NLLB model and the fine-tuned CTranslate2 model have their own licensing and attribution requirements. Refer to the respective model documentation before using the model commercially.
