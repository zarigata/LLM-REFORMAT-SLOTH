import os

def ensure_model_dir(model_id: str) -> str:
    outdir = os.path.join("models", model_id)
    os.makedirs(outdir, exist_ok=True)
    return outdir
