param(
  [Parameter(Mandatory=$true)][string]$ModelDir,
  [string]$Name='llm-factory-sample'
)

& (Join-Path $PSScriptRoot 'package_modelfile.ps1') -ModelDir $ModelDir
& (Join-Path $PSScriptRoot 'ollama_publish.ps1') -ModelDir $ModelDir -Name $Name
