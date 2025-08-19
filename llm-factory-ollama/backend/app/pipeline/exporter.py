import os
import time
from .utils import ensure_model_dir


def export_model(job, model_id: str, export_format: str = "gguf") -> str:
    outdir = ensure_model_dir(model_id)
    job.log(f"Export: format={export_format}")
    time.sleep(0.5)
    if export_format == "gguf":
        out = os.path.join(outdir, f"{model_id}.gguf")
    else:
        out = os.path.join(outdir, f"{model_id}.safetensors")
    with open(out, "wb") as f:
        f.write(b"EXPORT_PLACEHOLDER")
    job.artifacts["export"] = out
    job.progress = 90
    return out
