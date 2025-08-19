# Troubleshooting

- GPU not detected:
  - NVIDIA: `nvidia-smi` must succeed
  - AMD: `rocminfo` must succeed
  - Docker: ensure GPU pass-through enabled
- Out of memory:
  - Reduce LoRA rank or batch size
  - Use int8 quantization during training
  - Switch to CPU dry-run to validate flow
- Ollama create fails:
  - Inspect `models/<model_id>/Modelfile`
  - Test manually: `ollama create <name> -f models/<model_id>/Modelfile`
  - Ensure `ollama` service is running at `localhost:11434`
