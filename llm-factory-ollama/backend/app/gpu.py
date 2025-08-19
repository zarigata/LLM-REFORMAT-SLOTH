import shutil
import subprocess


def detect_gpu() -> dict:
    info = {"nvidia": False, "amd": False, "drivers": []}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            info["nvidia"] = True
            info["drivers"].append(out.splitlines()[0])
        except Exception as e:
            info["drivers"].append(f"nvidia-smi error: {e}")
    if shutil.which("rocminfo"):
        try:
            out = subprocess.check_output(["rocminfo"], stderr=subprocess.STDOUT, text=True)
            info["amd"] = True
            info["drivers"].append(out.splitlines()[0][:120])
        except Exception as e:
            info["drivers"].append(f"rocminfo error: {e}")
    return info
