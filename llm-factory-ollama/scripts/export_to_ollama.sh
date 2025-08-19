#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${1:-}
NAME=${2:-llm-factory-sample}
if [[ -z "$MODEL_DIR" ]]; then echo "Usage: $0 <model_dir> [ollama_name]" >&2; exit 1; fi
"$(dirname "$0")/package_modelfile.sh" "$MODEL_DIR"
"$(dirname "$0")/ollama_publish.sh" "$MODEL_DIR" "$NAME"
