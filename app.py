from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import ctranslate2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import NllbTokenizerFast


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")


MODEL_DIR = Path(
    os.getenv(
        "CT2_MODEL_DIR",
        "/mnt/d/aug_data/models/nllb_en_ml_runtime/nllb-1.3b-en-ml-ct2",
    )
).expanduser()


SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "eng_Latn")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "mal_Mlym")

CT2_DEVICE = os.getenv("CT2_DEVICE", "cpu")
CT2_COMPUTE_TYPE = os.getenv("CT2_COMPUTE_TYPE", "int8")

MAX_INPUT_CHARACTERS = int(
    os.getenv("MAX_INPUT_CHARACTERS", "5000")
)

MAX_INPUT_TOKENS = int(
    os.getenv("MAX_INPUT_TOKENS", "512")
)

MAX_BATCH_SIZE = int(
    os.getenv("MAX_BATCH_SIZE", "16")
)

MAX_CONCURRENT_GENERATIONS = int(
    os.getenv("MAX_CONCURRENT_GENERATIONS", "1")
)


tokenizer: NllbTokenizerFast | None = None
translator: ctranslate2.Translator | None = None

generation_gate = asyncio.Semaphore(
    MAX_CONCURRENT_GENERATIONS
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file not found: {path}")


def validate_model_files() -> None:
    if not MODEL_DIR.is_dir():
        raise RuntimeError(
            f"CTranslate2 model directory not found: {MODEL_DIR}"
        )

    require_file(MODEL_DIR / "model.bin")
    require_file(MODEL_DIR / "tokenizer.json")
    require_file(MODEL_DIR / "sentencepiece.bpe.model")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tokenizer, translator

    validate_model_files()

    print(f"Loading NLLB tokenizer from: {MODEL_DIR}")

    tokenizer = NllbTokenizerFast.from_pretrained(
        MODEL_DIR,
        src_lang=SOURCE_LANGUAGE,
        local_files_only=True,
        fix_mistral_regex=False,
    )

    print("Loading CTranslate2 model...")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Device: {CT2_DEVICE}")
    print(f"Compute type: {CT2_COMPUTE_TYPE}")

    translator = ctranslate2.Translator(
        str(MODEL_DIR),
        device=CT2_DEVICE,
        compute_type=CT2_COMPUTE_TYPE,
    )

    print("NLLB CTranslate2 model loaded successfully.")

    yield

    tokenizer = None
    translator = None


app = FastAPI(
    title="NLLB English–Malayalam Translator",
    version="2.0.0",
    description="FastAPI translation service using NLLB and CTranslate2.",
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(directory=APP_DIR / "static"),
    name="static",
)


class TranslateRequest(BaseModel):
    text: Annotated[
        str,
        Field(min_length=1, max_length=MAX_INPUT_CHARACTERS),
    ]

    beam_size: Annotated[
        int,
        Field(ge=1, le=8),
    ] = 4

    max_new_tokens: Annotated[
        int,
        Field(ge=8, le=512),
    ] = 256


class BatchTranslateRequest(BaseModel):
    texts: Annotated[
        list[str],
        Field(min_length=1, max_length=MAX_BATCH_SIZE),
    ]

    beam_size: Annotated[
        int,
        Field(ge=1, le=8),
    ] = 4

    max_new_tokens: Annotated[
        int,
        Field(ge=8, le=512),
    ] = 256


def translate_sync(
    texts: list[str],
    beam_size: int,
    max_new_tokens: int,
) -> tuple[list[str], int, int]:

    if tokenizer is None or translator is None:
        raise RuntimeError("Model is not loaded.")

    source_tokens = []

    input_token_count = 0

    for text in texts:

        encoded_ids = tokenizer.encode(
            text,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoded_ids
        )

        source_tokens.append(tokens)

        input_token_count += len(tokens)

    results = translator.translate_batch(
        source_tokens,
        target_prefix=[
            [TARGET_LANGUAGE]
            for _ in texts
        ],
        beam_size=beam_size,
        max_decoding_length=max_new_tokens,
    )

    translations = []

    output_token_count = 0

    for result in results:

        hypothesis = result.hypotheses[0]

        if hypothesis and hypothesis[0] == TARGET_LANGUAGE:
            hypothesis = hypothesis[1:]

        output_token_count += len(hypothesis)

        token_ids = tokenizer.convert_tokens_to_ids(
            hypothesis
        )

        translation = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations.append(
            translation.strip()
        )

    return (
        translations,
        input_token_count,
        output_token_count,
    )


async def perform_translation(
    texts: list[str],
    beam_size: int,
    max_new_tokens: int,
) -> tuple[list[str], int, int, float]:

    started = time.perf_counter()

    async with generation_gate:

        translations, input_tokens, output_tokens = (
            await run_in_threadpool(
                translate_sync,
                texts,
                beam_size,
                max_new_tokens,
            )
        )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    return (
        translations,
        input_tokens,
        output_tokens,
        elapsed_ms,
    )


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:

    return FileResponse(
        APP_DIR / "static" / "index.html"
    )


@app.get("/api/health")
async def health() -> dict:

    return {
        "status": (
            "ok"
            if translator is not None
            else "loading"
        ),
        "backend": "CTranslate2",
        "provider": (
            f"{CT2_DEVICE} / {CT2_COMPUTE_TYPE}"
        ),
        "source_language": SOURCE_LANGUAGE,
        "target_language": TARGET_LANGUAGE,
        "model_directory": str(MODEL_DIR),
        "model_format": "CTranslate2 INT8",
    }


@app.post("/api/translate")
async def translate(
    request: TranslateRequest,
) -> dict:

    source = request.text.strip()

    if not source:

        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty.",
        )

    try:

        (
            translations,
            input_tokens,
            output_tokens,
            elapsed_ms,
        ) = await perform_translation(
            [source],
            request.beam_size,
            request.max_new_tokens,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {exc}",
        ) from exc

    return {
        "source": source,
        "translation": translations[0],
        "source_language": SOURCE_LANGUAGE,
        "target_language": TARGET_LANGUAGE,
        "beam_size": request.beam_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_ms": round(
            elapsed_ms,
            2,
        ),
    }


@app.post("/api/translate-batch")
async def translate_batch(
    request: BatchTranslateRequest,
) -> dict:

    texts = [
        text.strip()
        for text in request.texts
    ]

    if any(not text for text in texts):

        raise HTTPException(
            status_code=400,
            detail="Batch inputs cannot contain empty strings.",
        )

    too_long = [
        index
        for index, text in enumerate(texts)
        if len(text) > MAX_INPUT_CHARACTERS
    ]

    if too_long:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Inputs at indexes {too_long} "
                "exceed the character limit."
            ),
        )

    try:

        (
            translations,
            input_tokens,
            output_tokens,
            elapsed_ms,
        ) = await perform_translation(
            texts,
            request.beam_size,
            request.max_new_tokens,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Batch translation failed: {exc}",
        ) from exc

    return {
        "items": [
            {
                "source": source,
                "translation": translation,
            }
            for source, translation
            in zip(texts, translations)
        ],
        "count": len(texts),
        "beam_size": request.beam_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_ms": round(
            elapsed_ms,
            2,
        ),
    }