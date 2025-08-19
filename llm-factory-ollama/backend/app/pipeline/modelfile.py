import os
from datetime import datetime


def write_modelfile(model_dir: str, model_path: str, base_model: str, params: dict | None = None) -> str:
    mf = os.path.join(model_dir, "Modelfile")
    meta = params or {}
    content = f"""
FROM {os.path.basename(model_path)}

# metadata
PARAMETER temperature 0.7
TEMPLATE "You are a helpful assistant."
LICENSE "MIT or Apache-2.0 (verify base model)"
# base_model: {base_model}
# date: {datetime.utcnow().isoformat()}Z
# hyperparams: {meta}
""".strip()
    with open(mf, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    return mf
