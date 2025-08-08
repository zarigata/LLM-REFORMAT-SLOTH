from __future__ import annotations
import os
from typing import Optional
from pydantic import BaseModel

from .jobs import Job
from .settings import settings


class FinetuneConfig(BaseModel):
    base_model_id: str
    dataset: str  # HF dataset name (e.g., 'yahma/alpaca-cleaned') or local path
    text_field: str = "text"
    output_dir: Optional[str] = None

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    max_seq_length: int = 2048

    use_4bit: bool = True
    bf16: bool = False


_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _load_dataset(job: Job, cfg: FinetuneConfig):
    from datasets import load_dataset
    import os

    if os.path.exists(cfg.dataset):
        job.log(f"Loading local dataset from: {cfg.dataset}")
        # Heuristic: try to load json/jsonl; otherwise, treat as text files
        if os.path.isdir(cfg.dataset):
            return load_dataset("text", data_dir=cfg.dataset)
        else:
            ext = os.path.splitext(cfg.dataset)[1].lower()
            if ext in {".json", ".jsonl"}:
                return load_dataset("json", data_files=cfg.dataset)
            return load_dataset("text", data_files=cfg.dataset)
    else:
        job.log(f"Loading HF dataset: {cfg.dataset}")
        return load_dataset(cfg.dataset)


def run_finetune(job: Job, cfg: FinetuneConfig) -> None:
    """Execute a LoRA/QLoRA fine-tune using Unsloth and TRL SFTTrainer."""
    job.log("Starting fine-tuning job")

    # Lazy imports to allow UI to start even if heavy deps missing
    try:
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except Exception as e:
        job.log("Failed to import training libraries. Ensure 'unsloth', 'transformers', 'trl' are installed.")
        raise

    ds = _load_dataset(job, cfg)

    output_dir = cfg.output_dir or os.path.join(settings.outputs_dir, "lora_" + cfg.base_model_id.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True)

    job.log(f"Loading base model: {cfg.base_model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model_id,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=bool(cfg.use_4bit),
        device_map="auto",
        token=None,  # uses HF token from cache if configured
    )

    job.log("Applying LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(_DEFAULT_TARGET_MODULES),
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
    )

    # Prepare dataset formatting
    text_field = cfg.text_field

    def format_example(ex):
        # Minimal: treat each row as plain text prompt-target in one field
        txt = ex.get(text_field)
        if isinstance(txt, list):
            txt = "\n".join([str(t) for t in txt])
        return {"text": str(txt)}

    job.log("Mapping dataset to text field")
    if "train" in ds:
        ds["train"] = ds["train"].map(format_example)
        train_dataset = ds["train"]
    else:
        ds = ds.map(format_example)
        train_dataset = ds["train"] if "train" in ds else ds

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=bool(cfg.bf16),
        fp16=not cfg.bf16,
        optim="paged_adamw_32bit",
        report_to=[],
    )

    job.log("Starting SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=True,
        args=args,
    )

    trainer.train()
    job.log("Training complete. Saving LoRA adapter and tokenizer.")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    job.log(f"Artifacts saved to: {output_dir}")
