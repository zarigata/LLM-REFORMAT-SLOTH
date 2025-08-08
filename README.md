# Unsloth LLM Fine-Tuner for Ollama

A Python web GUI to:
- Browse/search Hugging Face models and datasets
- Download models
- Fine-tune LLMs with Unsloth (LoRA/QLoRA)
- Build a Modelfile and create a model on an Ollama server

## Stack
- FastAPI + Gradio UI
- Unsloth + TRL + Transformers
- Hugging Face Hub client
- Ollama REST API (`/api/create`)

## Prerequisites
- Python 3.10+
- (Recommended) NVIDIA GPU + CUDA for training.
- (Optional) Docker — recommended on Windows for GPU & bitsandbytes stability.
- Hugging Face token for gated models (set `HUGGINGFACE_TOKEN`).
- An Ollama server reachable at `http://localhost:11434` or custom URL.

## Environment
Before setup, decide if you will use Docker or native Windows.

### Native Windows (PowerShell or cmd)
```powershell
# PowerShell
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```
```bat
:: cmd.exe
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```

Notes:
- On Windows, `bitsandbytes` may be limited. If install fails, you can remove it from `requirements.txt` and run in 8-bit/16-bit modes supported by your setup, or prefer Docker.
- Install PyTorch matching your CUDA as per https://pytorch.org/get-started/locally/

### Docker (optional)
A Dockerfile is not yet provided. If you want it, ask and we will generate one that supports CUDA.

## Usage
1. Open the app in your browser (default http://localhost:7860/).
2. Use the Models/Datasets tabs to search.
3. Download a model if needed (caches to `hf_cache/`).
4. Fine-tune tab: set base model id (e.g., `unsloth/llama-3-8b-bnb-4bit`) and dataset (`yahma/alpaca-cleaned` or local path). Start training — a job id is returned; view status/logs.
5. Ollama Export: build a minimal Modelfile (FROM + TEMPLATE) and send to your Ollama server via `/api/create`.

### Important about Ollama
- Running fine-tuned LoRA in Ollama typically requires merging adapters and converting to GGUF. This app generates and sends a Modelfile; you may need to convert/prepare weights externally before the model runs in Ollama. We can add an automated merge-and-convert step if desired.

## Configuration
- `HUGGINGFACE_TOKEN` — for private/gated repos
- `HF_HOME` — cache dir (default `./hf_cache`)
- `DATA_DIR`, `OUTPUTS_DIR` — data and outputs (default `./data`, `./outputs`)
- `OLLAMA_URL` — Ollama server URL (default `http://localhost:11434`)

## Limitations / Next steps
- Optional Dockerfile + compose for reproducibility and GPU
- Model card preview and file browser in UI
- Merge LoRA and auto-convert to GGUF for Ollama
- Inference/test playground tab
- Queue persistence across restarts
