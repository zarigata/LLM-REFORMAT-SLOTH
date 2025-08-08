import os
from fastapi import FastAPI
import gradio as gr
import uvicorn

from .ui import build_ui
from .device import get_device_info
from . import settings as app_settings


def create_app() -> FastAPI:
    app = FastAPI(title="Unsloth LLM Fine-Tuner for Ollama")
    gr_app = build_ui()
    gr.mount_gradio_app(app, gr_app, path="/")

    @app.get("/healthz")
    def healthz():
        try:
            dev = get_device_info()
        except Exception as e:
            dev = {"error": str(e)}
        s = app_settings.settings
        return {
            "status": "ok",
            "device": dev,
            "settings": {
                "OLLAMA_URL": s.ollama_url,
                "DATA_DIR": s.data_dir,
                "OUTPUTS_DIR": s.outputs_dir,
                "HF_HOME": s.hf_cache_dir,
                "HUGGINGFACE_TOKEN_set": bool(s.hf_token),
            },
        }
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
