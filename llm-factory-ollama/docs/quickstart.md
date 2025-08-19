# Quickstart

This guide uses Docker (CPU by default). GPU is optional.

## Windows (PowerShell/cmd)
```
scripts\quickstart.ps1
```

## Linux/macOS (bash/zsh)
```
./scripts/quickstart.sh
```

The script will:
- Validate Docker and (optionally) GPU drivers
- Build image `llm-factory-ollama:latest`
- Start `docker-compose` with the backend, UI, and optional Ollama service
- Open the UI at http://localhost:8000
- Run a dry-run pipeline to verify the flow

## GPU Setup
- NVIDIA:
  - Install latest drivers + CUDA toolkit (runtime only is fine)
  - Docker: enable GPU pass-through
    - Linux: `docker run --gpus all ...`
    - Windows: Docker Desktop → Settings → Resources → GPU
  - Test: `nvidia-smi`
- AMD (ROCm):
  - Install ROCm per your distro and GPU support
  - Test: `rocminfo` and `rocm-smi`

## Native Install
See `docs/runbook.md` for Python venv setup using system `python`, then:
```
python -m venv .venv
. .venv/Scripts/activate    # Windows PowerShell
# or
source .venv/bin/activate   # Linux/macOS
pip install -r backend/requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
```
