param(
  [Parameter(Mandatory=$true)][string]$ModelDir,
  [Parameter(Mandatory=$true)][string]$Name
)

if (-not (Test-Path (Join-Path $ModelDir 'Modelfile'))) {
  Write-Error "Modelfile not found in $ModelDir"; exit 1
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host 'ollama not found; ensure the sidecar is running at http://localhost:11434'
}

& ollama create $Name -f (Join-Path $ModelDir 'Modelfile')
Write-Host "Created $Name. To run: ollama run $Name"
