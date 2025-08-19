$ErrorActionPreference = 'Stop'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error 'Docker is required. Install Docker Desktop first.'
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Join-Path $ScriptDir '..' | Resolve-Path

# Build
docker build -f (Join-Path $Root 'docker/Dockerfile') -t llm-factory-ollama:latest $Root

# Compose up
Push-Location (Join-Path $Root 'docker')
docker compose up -d
Pop-Location

Write-Host 'UI: http://localhost:8000'

Write-Host 'Dry-run: starting sample job'
try {
  curl.exe -s -X POST http://localhost:8000/api/fine-tune -H 'Content-Type: application/json' -d '{"base_model_source":"huggingface","target_gpu":"cpu","dry_run":true}' | Out-Null
} catch {}
