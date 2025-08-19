import os
import time
from .utils import ensure_model_dir


def run_dpo(job, req: dict):
    job.log("[DPO] Starting direct preference optimization (dry-run)")
    model_id = f"dpo-{int(time.time())}"
    outdir = ensure_model_dir(model_id)
    steps = [
        "Loading base model",
        "Loading preference dataset (simulated)",
        "Computing preference loss",
        "Optimizing policy",
        "Saving adapters/policy",
    ]
    for i, step in enumerate(steps, 1):
        time.sleep(0.5)
        job.progress = int(i / len(steps) * 60)
        job.log(step)
    with open(os.path.join(outdir, "dpo_policy.safetensors"), "wb") as f:
        f.write(b"DPO_PLACEHOLDER")
    job.artifacts["policy"] = os.path.join(outdir, "dpo_policy.safetensors")
    job.artifacts["model_id"] = model_id
    job.log(f"Artifacts at {outdir}")
    job.progress = 60
    return model_id
