import os
from pydantic import BaseModel
from typing import Optional

from . import config_store


class Settings(BaseModel):
    """Centralized settings for the app. Values can be overridden via env vars or .env."""
    hf_token: Optional[str] = None
    hf_cache_dir: str = os.path.join(os.getcwd(), "hf_cache")
    data_dir: str = os.path.join(os.getcwd(), "data")
    outputs_dir: str = os.path.join(os.getcwd(), "outputs")
    ollama_url: str = "http://localhost:11434"


def _env_or_default(key: str, default: Optional[str]) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def _load_settings_from_sources() -> Settings:
    cfg = config_store.load_config()
    st = Settings(
        hf_token=_env_or_default("HUGGINGFACE_TOKEN", cfg.get("HUGGINGFACE_TOKEN")),
        hf_cache_dir=_env_or_default("HF_HOME", os.path.join(os.getcwd(), "hf_cache")),
        data_dir=_env_or_default("DATA_DIR", os.path.join(os.getcwd(), "data")),
        outputs_dir=_env_or_default("OUTPUTS_DIR", os.path.join(os.getcwd(), "outputs")),
        ollama_url=_env_or_default("OLLAMA_URL", cfg.get("OLLAMA_URL") or "http://localhost:11434"),
    )
    return st


settings = _load_settings_from_sources()


def apply_config(update: dict) -> Settings:
    """Persist config to file and refresh runtime settings. Returns the updated settings."""
    new_cfg = config_store.save_config(update)
    global settings
    settings = _load_settings_from_sources()
    # Ensure directories exist
    for _d in (settings.hf_cache_dir, settings.data_dir, settings.outputs_dir):
        try:
            os.makedirs(_d, exist_ok=True)
        except Exception:
            pass
    return settings


# Ensure base directories exist at startup
for _d in (settings.hf_cache_dir, settings.data_dir, settings.outputs_dir):
    try:
        os.makedirs(_d, exist_ok=True)
    except Exception:
        pass
