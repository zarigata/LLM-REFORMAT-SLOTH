import os
import time
from .utils import ensure_model_dir


def run_rlhf(job, req: dict):
    job.log("[RLHF] Starting preference collection and policy optimization (dry-run)")
    model_id = f"rlhf-{int(time.time())}"
    outdir = ensure_model_dir(model_id)
    steps = [
        "Loading base model",
        "Collecting preferences (simulated)",
        "Computing rewards (simulated)",
        "Policy optimization",
        "Saving policy",
    ]
    for i, step in enumerate(steps, 1):
        time.sleep(0.5)
        job.progress = int(i / len(steps) * 60)
        job.log(step)
    with open(os.path.join(outdir, "rlhf_policy.safetensors"), "wb") as f:
        f.write(b"RLHF_PLACEHOLDER")
    job.artifacts["policy"] = os.path.join(outdir, "rlhf_policy.safetensors")
    job.artifacts["model_id"] = model_id
    job.log(f"Artifacts at {outdir}")
    job.progress = 60
    return model_id
