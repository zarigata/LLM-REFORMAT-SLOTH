#!/usr/bin/env bash
set -euo pipefail

# Detect docker
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop/Engine first." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Build
docker build -f "$ROOT_DIR/docker/Dockerfile" -t llm-factory-ollama:latest "$ROOT_DIR"

# Compose up
(cd "$ROOT_DIR/docker" && docker compose up -d)

# Open UI hint
echo "UI: http://localhost:8000"

echo "Dry-run: starting sample job"
curl -s -X POST http://localhost:8000/api/fine-tune -H 'Content-Type: application/json' -d '{"base_model_source":"huggingface","target_gpu":"cpu","dry_run":true}' || true
