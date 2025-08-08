from __future__ import annotations
import json
import os
from typing import Any, Dict

CONFIG_PATH = os.path.join(os.getcwd(), "config.json")


_DEFAULTS: Dict[str, Any] = {
    "HUGGINGFACE_TOKEN": None,
    "OLLAMA_URL": "http://localhost:11434",
}


def load_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                if isinstance(file_cfg, dict):
                    cfg.update(file_cfg)
    except Exception:
        pass
    return cfg


def save_config(update: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config()
    cfg.update({k: v for k, v in update.items() if v is not None})
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return cfg
