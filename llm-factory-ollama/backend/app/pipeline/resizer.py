import os
import time


def resize(job, model_dir: str, settings: dict | None):
    if not settings:
        return None
    job.log(f"Resizer: {settings}")
    time.sleep(0.5)
    marker = os.path.join(model_dir, "resize.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(str(settings))
    job.progress = 70
    return marker
