#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${1:-}
NAME=${2:-}
if [[ -z "$MODEL_DIR" || -z "$NAME" ]]; then
  echo "Usage: $0 <model_dir> <ollama_name>" >&2; exit 1; fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found; ensure sidecar service is running on :11434" >&2
fi

ollama create "$NAME" -f "$MODEL_DIR/Modelfile"
echo "Created $NAME. To run: ollama run $NAME"
