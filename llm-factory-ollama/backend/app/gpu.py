import shutil
import subprocess
from typing import Any, Dict, List


def detect_gpu() -> dict:
    info: Dict[str, Any] = {"nvidia": False, "amd": False, "drivers": [], "gpus": []}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            info["nvidia"] = True
            info["drivers"].append(out.splitlines()[0])
            # Query structured info
            try:
                q = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ], text=True)
                for line in q.strip().splitlines():
                    name, mem_total, mem_free, drv = [x.strip() for x in line.split(",")]
                    info["gpus"].append({
                        "vendor": "nvidia",
                        "name": name,
                        "memory_total_mb": int(mem_total),
                        "memory_free_mb": int(mem_free),
                        "driver": drv,
                    })
            except Exception as e:
                info["drivers"].append(f"nvidia-smi query error: {e}")
        except Exception as e:
            info["drivers"].append(f"nvidia-smi error: {e}")
    if shutil.which("rocminfo"):
        try:
            out = subprocess.check_output(["rocminfo"], stderr=subprocess.STDOUT, text=True)
            info["amd"] = True
            info["drivers"].append(out.splitlines()[0][:120])
        except Exception as e:
            info["drivers"].append(f"rocminfo error: {e}")
    # Try rocm-smi for AMD VRAM
    if shutil.which("rocm-smi"):
        try:
            pname = subprocess.check_output(["rocm-smi", "--showproductname"], text=True, stderr=subprocess.STDOUT)
            vram = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], text=True, stderr=subprocess.STDOUT)
            # Very rough parse: collect lines with "GPU" and numbers in MiB
            lines = vram.splitlines()
            names = [l.split(":")[-1].strip() for l in pname.splitlines() if "card" in l.lower() or "GPU" in l]
            mems: List[Dict[str, Any]] = []
            for l in lines:
                if "VRAM Total" in l or "Total VRAM" in l:
                    try:
                        val = int("".join([c for c in l if c.isdigit()]))
                        mems.append({"memory_total_mb": val})
                    except Exception:
                        pass
                if "VRAM Used" in l or "VRAM in use" in l:
                    try:
                        val = int("".join([c for c in l if c.isdigit()]))
                        if mems:
                            mems[-1]["memory_used_mb"] = val
                    except Exception:
                        pass
            for i, m in enumerate(mems):
                info["gpus"].append({
                    "vendor": "amd",
                    "name": names[i] if i < len(names) else "AMD GPU",
                    **m,
                })
        except Exception as e:
            info["drivers"].append(f"rocm-smi error: {e}")
    return info


def diagnose_gpu(max_chars: int = 4000) -> Dict[str, str]:
    cmds = {
        "nvidia-smi": ["nvidia-smi"],
        "rocminfo": ["rocminfo"],
        "rocm-smi": ["rocm-smi", "--showall"],
    }
    res: Dict[str, str] = {}
    for name, cmd in cmds.items():
        if shutil.which(cmd[0]):
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
                res[name] = out[:max_chars]
            except Exception as e:
                res[name] = f"error: {e}"
        else:
            res[name] = "not found"
    return res
