from __future__ import annotations
from typing import Optional
from . import settings as app_settings


def generate_text(
    base_model_id: str,
    prompt: str,
    adapter_dir: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_4bit: bool = True,
    bf16: bool = False,
    max_seq_length: int = 4096,
) -> str:
    """Simple text generation with optional LoRA adapter.
    Tries 4-bit loading when requested; falls back if unavailable.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        return f"Missing libs (torch/transformers/peft): {e}"

    dtype = torch.bfloat16 if bf16 else None

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    if use_4bit:
        # If bitsandbytes not available or CPU-only, this may fail; we handle below.
        load_kwargs.update({
            "load_in_4bit": True,
        })

    token = app_settings.settings.hf_token

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_id, token=token, **load_kwargs)
    except Exception as e:
        # Fallback without 4bit
        if use_4bit:
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model_id, token=token, device_map="auto", torch_dtype=dtype)
            except Exception as e2:
                return f"Failed to load model: {e2} (original 4bit error: {e})"
        else:
            return f"Failed to load model: {e}"

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, token=token)

    if adapter_dir:
        try:
            model = PeftModel.from_pretrained(model, adapter_dir)
        except Exception as e:
            return f"Failed to load adapter from {adapter_dir}: {e}"

    model.eval()

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens or 256),
        do_sample=True,
        temperature=float(temperature or 0.7),
        top_p=float(top_p or 0.9),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True)
    return text
