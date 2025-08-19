# LLM Factory → Ollama Packager

A local, zero-code workflow to:
- Select a base LLM (local/HF/GitHub/URL)
- Fine-tune via LoRA/PEFT (default) or full finetune, optional RLHF/DPO
- Quantize/resize/stack adapters
- Export to GGUF or safetensors
- Auto-generate an Ollama Modelfile and run `ollama create`
- Serve via `ollama serve` with a simple REST UI
- Run everything inside one Docker container (or native install)

Licenses: MIT/Apache-2.0 only by default. Non-permissive models prompt a warning.

## Quick Start (Docker, CPU default)

- Windows PowerShell or cmd:
```
# optional: ensure Docker Desktop is running
scripts\quickstart.ps1
```
- Linux/macOS:
```
./scripts/quickstart.sh
```
This performs a dry-run (no downloads), builds the image, runs docker-compose, opens the web UI, and shows how to publish to Ollama.

## Requirements
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Optional GPU drivers:
  - NVIDIA: CUDA drivers + `--gpus all` in Docker
  - AMD: ROCm drivers; see docs for compose flags
- Native install path is documented in `docs/quickstart.md`

## Structure
- `backend/` FastAPI service with job runner and pipelines
- `frontend/` Accessible, responsive wizard (vanilla JS)
- `docker/` Dockerfile + docker-compose
- `ci/` GitHub Actions workflows under `.github/workflows/`
- `docs/` Quickstart, Runbook, Troubleshooting
- `models/` Exported artifacts
- `scripts/` Helpers: publish to Ollama, packaging, changelog

## Security & Privacy
- Data stays local by default. Nothing is pushed remotely without explicit consent.
- Optional encryption for artifacts can be enabled (placeholder in exporter).

## Accessibility
The UI targets WCAG AA, supports keyboard navigation and prefers-reduced-motion.

## License
MIT. See `LICENSE`.
