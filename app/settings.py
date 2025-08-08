import os
from pydantic import BaseModel


class Settings(BaseModel):
    """Centralized settings for the app. Values can be overridden via env vars or .env."""
    hf_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
    hf_cache_dir: str = os.getenv("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
    data_dir: str = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    outputs_dir: str = os.getenv("OUTPUTS_DIR", os.path.join(os.getcwd(), "outputs"))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")


settings = Settings()

# Ensure directories exist
for _d in (settings.hf_cache_dir, settings.data_dir, settings.outputs_dir):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass
