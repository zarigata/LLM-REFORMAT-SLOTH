from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .config import FineTuneRequest, ExportRequest, PublishRequest
from .jobs import manager
from .pipeline.finetune_lora import run_lora
from .pipeline.rlhf import run_rlhf
from .pipeline.dpo import run_dpo
from .pipeline.quantization import quantize
from .pipeline.resizer import resize
from .pipeline.exporter import export_model
from .pipeline.modelfile import write_modelfile
from .pipeline.utils import ensure_model_dir
from . import gpu as gpu_utils
import os
import shutil
import subprocess

app = FastAPI(title="LLM Factory → Ollama Packager")

# Serve frontend (copied into backend/public or from ../frontend)
if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "public")):
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public"))
else:
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")


@app.post("/api/fine-tune")
def start_finetune(req: FineTuneRequest):
    if not req.base_model_source or not req.target_gpu:
        raise HTTPException(status_code=400, detail="Missing required inputs: base_model_source, target_gpu")

    def task(job):
        job.log("Job started: fine-tune")
        model_id = None
        if req.fine_tune_method == "lora":
            model_id = run_lora(job, req.model_dump())
        elif req.fine_tune_method == "rlhf":
            model_id = run_rlhf(job, req.model_dump())
        elif req.fine_tune_method == "dpo":
            model_id = run_dpo(job, req.model_dump())
        elif req.fine_tune_method == "full_finetune":
            job.log("Full finetune not implemented; using LoRA fallback")
            model_id = run_lora(job, req.model_dump())
        else:
            job.log(f"Unknown method {req.fine_tune_method}; using LoRA fallback")
            model_id = run_lora(job, req.model_dump())
        # optional transforms
        resize(job, os.path.join("models", model_id), req.resizer_settings)
        quantize(job, os.path.join("models", model_id), req.quantization_target)
        job.log("Fine-tune complete")
        job.progress = 80

    job_id = manager.create("fine-tune", req.model_dump(), task)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "logs_tail": job.logs[-20:],
        "artifacts": job.artifacts,
        "error": job.error,
    }


@app.post("/api/export/{model_id}")
def do_export(model_id: str, req: ExportRequest):
    model_dir = ensure_model_dir(model_id)
    if not os.path.isdir(model_dir):
        raise HTTPException(status_code=404, detail="Model dir not found")

    def task(job):
        p = export_model(job, model_id, req.export_format)
        mf = write_modelfile(model_dir, p, base_model="unknown", params={"export_format": req.export_format})
        job.artifacts["modelfile"] = mf
        job.log("Export complete")
        job.progress = 100

    job_id = manager.create("export", req.model_dump(), task)
    return {"job_id": job_id}


@app.post("/api/ollama/publish/{model_id}")
def publish_ollama(model_id: str, req: PublishRequest):
    model_dir = ensure_model_dir(model_id)
    mf = os.path.join(model_dir, "Modelfile")
    if not os.path.isfile(mf):
        raise HTTPException(status_code=400, detail="Modelfile missing; run export first")

    def task(job):
        if shutil.which("ollama") is None:
            job.log("ollama not found; attempting to call sidecar at 11434")
        # 'ollama create' will use the Modelfile
        try:
            subprocess.check_call(["ollama", "create", req.ollama_name, "-f", mf])
            job.artifacts["ollama_name"] = req.ollama_name
            job.artifacts["serve_url"] = "http://localhost:11434"
            job.log("Ollama create success")
        except Exception as e:
            job.log(f"Ollama create failed: {e}")
            job.error = str(e)
            # Still provide manual instructions
        job.progress = 100

    job_id = manager.create("ollama_publish", req.model_dump(), task)
    return {"job_id": job_id}


@app.get("/api/metrics/gpu")
def metrics_gpu():
    return gpu_utils.detect_gpu()


@app.get("/api/artifact/{model_id}/{fname}")
def download_artifact(model_id: str, fname: str):
    path = os.path.join("models", model_id, fname)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@app.get("/api/summary/{model_id}")
def build_summary(model_id: str):
    model_dir = os.path.join("models", model_id)
    export_path = None
    for cand in (f"{model_id}.gguf", f"{model_id}.safetensors"):
        p = os.path.join(model_dir, cand)
        if os.path.isfile(p):
            export_path = p
            break
    res = {
        "success": bool(export_path),
        "model_id": model_id,
        "export": export_path,
        "ollama": {"created": False, "name": None, "serve_url": "http://localhost:11434"},
        "ollama_status": "unknown",
        "git_commit": {"sha": None, "push_url": None},
        "git_diff_url": None,
        "changelog_summary": None,
        "artifacts_paths": [model_dir],
    }
    return JSONResponse(res)
