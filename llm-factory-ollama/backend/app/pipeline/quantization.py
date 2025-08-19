import os
import time


def quantize(job, model_dir: str, method: str = "int8"):
    job.log(f"Quantization: {method}")
    time.sleep(0.5)
    # Placeholder; in real run, invoke GPTQ/AWQ tools here
    marker = os.path.join(model_dir, f"quant_{method}.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("quantized")
    job.progress = 75
    return marker
