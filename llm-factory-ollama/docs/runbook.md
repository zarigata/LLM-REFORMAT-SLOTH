# Runbook

## Inputs
- base_model_source: local_path | huggingface | github_release | url
- target_gpu: nvidia | amd | cpu
- fine_tune_method: lora | full_finetune | rlhf | dpo (default lora)
- dataset: CSV/JSONL/TXT or dataset repo URL
- lora_params: r, alpha, dropout, adapter_size
- quantization_target: none | int8 | int4 | fp8 (default int8)
- resizer_settings: shrink_or_expand, percent, preserve_layers
- export_format: gguf | safetensors (default gguf)
- ollama_target: name, auto_serve
- ui_preferences: theme, accessibility, language
- docker_mode: container_all_in_one | container_backend_only | native_install

## API Endpoints
- `POST /api/fine-tune` start job (supports dry_run)
- `GET /api/status/{job_id}` check progress
- `POST /api/export/{model_id}` export artifacts
- `POST /api/ollama/publish/{model_id}` create in Ollama
- `GET /api/metrics/gpu` GPU diagnostics

## Logs and Artifacts
- Per-job logs under `logs/{job_id}.log`
- Models under `models/{model_id}/`
- Modelfile generated at `models/{model_id}/Modelfile`

## Error Handling
- GPU/driver issues: see Troubleshooting, use Diagnose button in UI
- Export failures keep intermediates and stacktraces
- Ollama create failures produce a downloadable Modelfile with manual steps
