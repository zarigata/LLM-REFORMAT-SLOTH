from __future__ import annotations
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "backend": "cpu",
        "num_gpus": 0,
        "gpu_names": [],
        "bf16_supported": False,
        "notes": [],
    }
    try:
        import torch
        # ROCm detection
        rocm = getattr(torch.version, "hip", None)
        cuda_ok = torch.cuda.is_available()
        if rocm:
            info["backend"] = "rocm"
        elif cuda_ok:
            info["backend"] = "cuda"
        else:
            # Try DirectML
            try:
                import torch_directml as _tdml  # type: ignore
                d = _tdml.device()
                info["backend"] = "directml"
                info["notes"].append("torch-directml available")
            except Exception:
                pass
        if info["backend"] in ("cuda", "rocm"):
            try:
                n = torch.cuda.device_count()
                info["num_gpus"] = n
                for i in range(n):
                    try:
                        info["gpu_names"].append(torch.cuda.get_device_name(i))
                    except Exception:
                        info["gpu_names"].append(f"GPU {i}")
            except Exception:
                pass
            try:
                info["bf16_supported"] = torch.cuda.is_bf16_supported()
            except Exception:
                # Not available on some builds
                pass
    except Exception as e:
        info["notes"].append(f"torch not available: {e}")
    return info
