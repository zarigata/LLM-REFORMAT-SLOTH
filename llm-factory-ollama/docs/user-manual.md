# LLM Factory → Ollama: User Manual (1 page)

This tool lets you fine-tune an open-source LLM and publish it to Ollama in a few clicks.

## 1) Start the app
- Easiest: Docker (CPU default)
  - Windows (PowerShell/cmd): `scripts\quickstart.ps1`
  - Linux/macOS: `./scripts/quickstart.sh`
- Opens UI at http://localhost:8000

Optional Native (advanced): see `docs/runbook.md`.

## 2) Pick settings in the Wizard
- Base model source: Hugging Face (default), Local path, GitHub release, URL
- Target hardware: CPU, NVIDIA, or AMD
- Method: LoRA (default), Full finetune, RLHF, DPO
- Export format: GGUF (default) or safetensors
- Dry-run: keep ON for a quick simulation (no downloads)

Click “Fine-Tune”. Watch logs update live.

## 3) Export and publish
- When training finishes, click “Export” to produce an artifact and Modelfile.
- Click “Publish to Ollama”. If Ollama is running (sidecar in compose), it creates the model.

Run locally with curl (example):
```
curl http://localhost:11434/api/generate -d '{"model":"llm-factory-sample","prompt":"Hello!"}'
```

## 4) Troubleshooting
- GPU not detected: Check drivers. NVIDIA: `nvidia-smi`, AMD: `rocminfo`.
- Out of memory: lower LoRA rank/batch or choose CPU dry-run.
- Ollama create fails: Download the `Modelfile` from the UI or `models/<id>/Modelfile` and run:
  - `ollama create <name> -f models/<id>/Modelfile`

## 5) Notes on licensing and privacy
- Default flow uses permissive OSS only (MIT/Apache-2.0). If you pick a non-permissive base model, the UI shows a warning.
- All data stays local unless you explicitly push to a remote.

## 6) Where are my files?
- Logs: `logs/<job_id>.log`
- Models and exports: `models/<model_id>/`
- Modelfile: `models/<model_id>/Modelfile`

## 7) CI/CD
- On PR: lint, tests, security scan, Docker build (no push)
- On merge to main: changelog update and automatic tag every 2 merged PRs (patch bump); can force minor.
