from __future__ import annotations
import requests
from typing import Optional


def create_model(ollama_url: str, model_name: str, modelfile: str) -> dict:
    """Create or update a model on an Ollama server by sending a Modelfile.
    Returns server response dict.
    """
    url = ollama_url.rstrip("/") + "/api/create"
    resp = requests.post(url, json={"name": model_name, "modelfile": modelfile}, timeout=600)
    resp.raise_for_status()
    return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"status_code": resp.status_code, "text": resp.text}
