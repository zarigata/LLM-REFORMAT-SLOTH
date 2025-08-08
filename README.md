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
- (Recommended) NVIDIA GPU + CUDA for training, or AMD ROCm on Linux, or DirectML on Windows.
- (Optional) Docker — recommended on Windows for GPU & bitsandbytes stability.
- Hugging Face token for gated models (you can set it in the app Settings tab).
- An Ollama server reachable at `http://localhost:11434` or custom URL.

## Environment
Choose one way to run:

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
- On Windows, `bitsandbytes` may be limited. If install fails, remove it from `requirements.txt` or use Docker.
- Install PyTorch matching your CUDA via https://pytorch.org/get-started/locally/
- Optional: for AMD/Intel on Windows, you can try `pip install torch-directml` and the app will detect DirectML.

### Docker (CPU)
Build and run a CPU-only container:

```bash
docker compose build app-cpu
docker compose up app-cpu
```

Mounts (created on first run): `./data`, `./outputs`, `./hf_cache`, and `./config.json`.

### Docker (CUDA GPU)
Requires NVIDIA GPU, drivers, and Docker GPU support enabled.

```bash
docker compose build app-cuda
docker compose up app-cuda
```

The CUDA image pre-installs PyTorch with cu121 wheels, then installs app deps.

## Usage
1. Open the app in your browser (default http://localhost:7860/).
2. Open the Settings tab:
   - Enter your Hugging Face token (not displayed back; stored in `config.json`).
   - Confirm/update the Ollama URL (default `http://localhost:11434`).
   - Review Device Info (CUDA/ROCm/DirectML/CPU).
3. Use the Models/Datasets tabs to search; download if needed (caches to `hf_cache/`).
4. Fine-tune tab: set base model id (e.g., `unsloth/llama-3-8b-bnb-4bit`) and dataset (`yahma/alpaca-cleaned` or local path). Start training — a job id is returned; view status/logs in Jobs.
5. Merge (LoRA→HF): merge adapter into base and save under `outputs/`.
6. Inference: quick text generation using base or base+adapter.
7. Ollama Export: build a minimal Modelfile (FROM + TEMPLATE) and send to your Ollama server via `/api/create`.

### Important about Ollama
- Running fine-tuned LoRA in Ollama typically requires merging adapters and converting to GGUF.
- This app can merge LoRA; for GGUF conversion use llama.cpp tooling (can be added later).

## Configuration
- Settings are persisted to `config.json` (HF token and Ollama URL). Env vars still override on startup:
  - `HUGGINGFACE_TOKEN`, `OLLAMA_URL`, `HF_HOME`, `DATA_DIR`, `OUTPUTS_DIR`
- Directories: `./hf_cache`, `./data`, `./outputs` are created if missing.

## Device Detection
- The app auto-detects: CUDA (NVIDIA), ROCm (AMD on Linux), DirectML (Windows), else CPU.
- Training/inference attempt 4-bit when enabled; falls back automatically if unsupported.

## Limitations / Next steps
- Optional GGUF conversion automation for Ollama
- Model card preview and file browser in UI
- Queue persistence across restarts
- Harden container images as needed
- Inference/test playground tab
- Queue persistence across restarts
