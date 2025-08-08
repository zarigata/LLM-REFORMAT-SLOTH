from __future__ import annotations
import requests
from typing import Optional
import time


def create_model(ollama_url: str, model_name: str, modelfile: str) -> dict:
    """Create or update a model on an Ollama server by sending a Modelfile.
    Returns server response dict.
    """
    url = ollama_url.rstrip("/") + "/api/create"
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = requests.post(url, json={"name": model_name, "modelfile": modelfile}, timeout=600)
            # Retry on 5xx
            if 500 <= resp.status_code < 600:
                last_err = RuntimeError(f"server {resp.status_code}: {resp.text[:200]}")
                raise last_err
            resp.raise_for_status()
            return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"status_code": resp.status_code, "text": resp.text}
        except Exception as e:
            last_err = e
            # Exponential backoff
            sleep_s = min(2 ** attempt, 10)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to create model after retries: {last_err}")
