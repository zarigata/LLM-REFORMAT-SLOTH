from pydantic import BaseModel
from typing import Optional, Literal

class FineTuneRequest(BaseModel):
    base_model_source: Literal["local_path", "huggingface", "github_release", "url"] = "huggingface"
    base_model: Optional[str] = "Qwen/Qwen2.5-0.5B"  # small default
    target_gpu: Literal["nvidia", "amd", "cpu"] = "cpu"
    fine_tune_method: Literal["lora", "full_finetune", "rlhf", "dpo"] = "lora"
    dataset: Optional[str] = None
    lora_params: Optional[dict] = None
    quantization_target: Literal["none", "int8", "int4", "fp8"] = "int8"
    resizer_settings: Optional[dict] = None
    export_format: Literal["gguf", "safetensors"] = "gguf"
    ollama_target: Optional[dict] = None
    ui_preferences: Optional[dict] = None
    docker_mode: Literal["container_all_in_one", "container_backend_only", "native_install"] = "container_all_in_one"
    dry_run: bool = True

class ExportRequest(BaseModel):
    model_id: str
    export_format: Literal["gguf", "safetensors"] = "gguf"

class PublishRequest(BaseModel):
    model_id: str
    ollama_name: str
    auto_serve: bool = True
