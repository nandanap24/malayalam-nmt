# NLLB ONNX FastAPI Translator

Local English-to-Malayalam translation using the working ONNX export.

## Existing model structure

```text
~/models/nllb_en_ml_runtime/
├── .venv/
├── merged_hf/
└── onnx/
    ├── encoder_model.onnx
    ├── decoder_model.onnx
    └── decoder_with_past_model.onnx
```

The NLLB tokenizer is loaded from `merged_hf`. The ONNX model is loaded with:

```python
use_cache=True
use_merged=False
```

## Copy and install

```bash
cd ~/models/nllb_en_ml_runtime
unzip ~/Downloads/nllb_onnx_fastapi_ui.zip
cd nllb_onnx_fastapi_ui

chmod +x install.sh run.sh
./install.sh
```

Verify `.env`:

```bash
nano .env
```

## Start

```bash
./run.sh
```

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## API test

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Artificial intelligence can improve language technology.",
    "beam_size": 4,
    "max_new_tokens": 256
  }'
```

## LAN access

The server binds to `0.0.0.0`. Find the machine IP:

```bash
hostname -I
```

Then open:

```text
http://MACHINE_IP:8000
```

Only one generation request is processed at a time by default to limit CPU and
memory pressure.
