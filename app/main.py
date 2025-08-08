import os
from fastapi import FastAPI
import gradio as gr
import uvicorn

from .ui import build_ui


def create_app() -> FastAPI:
    app = FastAPI(title="Unsloth LLM Fine-Tuner for Ollama")
    gr_app = build_ui()
    gr.mount_gradio_app(app, gr_app, path="/")
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
