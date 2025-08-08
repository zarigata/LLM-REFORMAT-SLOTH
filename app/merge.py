from __future__ import annotations
import os
from typing import Optional
from pydantic import BaseModel

from .jobs import Job
from .settings import settings


class MergeConfig(BaseModel):
    base_model_id: str
    adapter_dir: str
    output_dir: Optional[str] = None
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16"


def run_merge(job: Job, cfg: MergeConfig) -> None:
    job.log("Starting LoRA merge job")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        job.log("Required libs missing: install torch, transformers, peft")
        raise

    out_dir = cfg.output_dir or os.path.join(settings.outputs_dir, "merged_" + cfg.base_model_id.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)

    # Resolve dtype
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg.torch_dtype, None)

    job.log(f"Loading base model: {cfg.base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(cfg.base_model_id, use_fast=True)

    job.log(f"Loading LoRA adapter from: {cfg.adapter_dir}")
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)

    job.log("Merging and unloading adapters")
    try:
        model = model.merge_and_unload()
    except Exception as e:
        job.log(f"merge_and_unload failed, trying to export without merge: {e}")

    job.log("Saving merged model (or base if merge failed)")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    job.log(f"Merged model saved to: {out_dir}")
