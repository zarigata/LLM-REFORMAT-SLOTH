import os
import time
from .utils import ensure_model_dir


def run_lora(job, req: dict):
    # Dry-run friendly: simulate steps with logs and generate tiny artifact
    model_id = f"model-{int(time.time())}"
    outdir = ensure_model_dir(model_id)

    job.log("[LoRA] Starting fine-tune (dry-run=" + str(req.get("dry_run", True)) + ")")
    steps = [
        "Loading base model",
        "Preparing dataset",
        "Attaching LoRA adapters",
        "Training loop",
        "Saving adapters",
    ]
    for i, step in enumerate(steps, 1):
        time.sleep(0.5)
        job.progress = int(i / len(steps) * 60)
        job.log(step)

    # Create a tiny placeholder adapter file
    with open(os.path.join(outdir, "lora_adapter.safetensors"), "wb") as f:
        f.write(b"SFT_PLACEHOLDER")
    job.artifacts["adapter"] = os.path.join(outdir, "lora_adapter.safetensors")
    job.artifacts["model_id"] = model_id
    job.log(f"Artifacts at {outdir}")
    job.progress = 60
    return model_id
