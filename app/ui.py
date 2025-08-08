from __future__ import annotations
import os
from typing import List

import gradio as gr
import pandas as pd

from . import hf_client
from .jobs import create_job, get_job, list_jobs, run_in_thread
from .finetune import FinetuneConfig, run_finetune
from .ollama_client import create_model as ollama_create_model
from .settings import settings, apply_config
from . import fs_utils
from . import infer
from .merge import MergeConfig, run_merge
from .device import get_device_info


def _search_models(query, limit, task, library, sort, direction):
    rows = hf_client.search_models(query or "", int(limit or 20), task or None, library or None, sort or None, direction or None)
    return pd.DataFrame(rows)


def _search_datasets(query, limit, task, sort, direction):
    rows = hf_client.search_datasets(query or "", int(limit or 20), task or None, sort or None, direction or None)
    return pd.DataFrame(rows)


def _download_model(repo_id):
    if not repo_id:
        return "Repo id required"
    path = hf_client.download_model(repo_id)
    return f"Downloaded to: {path}"


def _model_readme(repo_id):
    if not repo_id:
        return "Enter a model repo id"
    return hf_client.get_model_readme(str(repo_id).strip())


def _dataset_readme(repo_id):
    if not repo_id:
        return "Enter a dataset id"
    return hf_client.get_dataset_readme(str(repo_id).strip())


def _upload_to_data(files):
    import shutil
    saved = []
    if not files:
        return "No files selected"
    os.makedirs(settings.data_dir, exist_ok=True)
    for f in (files if isinstance(files, list) else [files]):
        # Gradio may pass temp file path or object with name attribute
        src = getattr(f, "name", None) or getattr(f, "orig_name", None) or str(f)
        if not src:
            continue
        base = os.path.basename(src)
        dst = os.path.join(settings.data_dir, base)
        try:
            shutil.copy2(src, dst)
            saved.append(dst)
        except Exception as e:
            saved.append(f"Failed: {base}: {e}")
    return "\n".join(saved)


def _list_data_dir():
    rows = fs_utils.list_dir_tree(settings.data_dir)
    return pd.DataFrame(rows)


def _list_outputs_dir():
    rows = fs_utils.list_dir_tree(settings.outputs_dir)
    return pd.DataFrame(rows)


def _start_merge(base_model_id, adapter_dir, out_dir, torch_dtype):
    cfg = MergeConfig(
        base_model_id=str(base_model_id).strip(),
        adapter_dir=str(adapter_dir).strip(),
        output_dir=str(out_dir).strip() or None,
        torch_dtype=str(torch_dtype or "auto").strip(),
    )
    job = create_job("merge", cfg.model_dump())

    def _task(j):
        try:
            run_merge(j, cfg)
        except Exception as e:
            j.log(f"Failed: {e}")
            j.status = "failed"

    run_in_thread(job, _task)
    return job.id


def _infer(base_model_id, prompt, adapter_dir, max_new_tokens, temperature, top_p, use_4bit, bf16, max_seq_length):
    text = infer.generate_text(
        base_model_id=str(base_model_id).strip(),
        prompt=str(prompt or "").strip(),
        adapter_dir=str(adapter_dir or "").strip() or None,
        max_new_tokens=int(max_new_tokens or 256),
        temperature=float(temperature or 0.7),
        top_p=float(top_p or 0.9),
        use_4bit=bool(use_4bit),
        bf16=bool(bf16),
        max_seq_length=int(max_seq_length or 4096),
    )
    return text


def _start_finetune(base_model_id, dataset, text_field, out_dir, lora_r, lora_alpha, lora_dropout,
                    learning_rate, per_device_train_batch_size, gradient_accumulation_steps, num_train_epochs,
                    max_seq_length, use_4bit, bf16):
    cfg = FinetuneConfig(
        base_model_id=str(base_model_id).strip(),
        dataset=str(dataset).strip(),
        text_field=str(text_field or "text").strip(),
        output_dir=str(out_dir).strip() or None,
        lora_r=int(lora_r or 16),
        lora_alpha=int(lora_alpha or 32),
        lora_dropout=float(lora_dropout or 0.05),
        learning_rate=float(learning_rate or 2e-4),
        per_device_train_batch_size=int(per_device_train_batch_size or 1),
        gradient_accumulation_steps=int(gradient_accumulation_steps or 8),
        num_train_epochs=float(num_train_epochs or 1.0),
        max_seq_length=int(max_seq_length or 2048),
        use_4bit=bool(use_4bit),
        bf16=bool(bf16),
    )
    job = create_job("finetune", cfg.model_dump())

    def _task(j):
        try:
            run_finetune(j, cfg)
        except Exception as e:
            j.log(f"Failed: {e}")
            j.status = "failed"

    run_in_thread(job, _task)
    return job.id


def _job_status(job_id):
    job = get_job(str(job_id))
    if not job:
        return "not-found", ""
    return job.status, "\n".join(job.logs[-500:])


def _list_jobs():
    rows = list_jobs()
    return pd.DataFrame(rows)


def _create_ollama_model(ollama_url, model_name, from_base, template_text):
    if not model_name:
        return {"error": "Model name required"}
    from_base = (from_base or "").strip()
    modelfile = f"FROM {from_base or 'llama3'}\n"
    if template_text and template_text.strip():
        modelfile += f"TEMPLATE \"{template_text.strip()}\"\n"
    try:
        res = ollama_create_model(str(ollama_url or settings.ollama_url), str(model_name), modelfile)
        return res
    except Exception as e:
        return {"error": str(e)}


def _get_settings():
    # Do not echo token back blindly; indicate if set.
    return {
        "HUGGINGFACE_TOKEN_set": bool(settings.hf_token),
        "OLLAMA_URL": settings.ollama_url,
        "DATA_DIR": settings.data_dir,
        "OUTPUTS_DIR": settings.outputs_dir,
        "HF_HOME": settings.hf_cache_dir,
    }


def _save_settings(hf_token, ollama_url, clear_token=False):
    update = {}
    if clear_token:
        update["HUGGINGFACE_TOKEN"] = None
    elif hf_token is not None and str(hf_token).strip() != "":
        update["HUGGINGFACE_TOKEN"] = str(hf_token).strip()
    if ollama_url is not None and str(ollama_url).strip() != "":
        update["OLLAMA_URL"] = str(ollama_url).strip()
    if not update:
        return "Nothing to update"
    apply_config(update)
    return "Saved settings. Restart not required."


def _device_info():
    return get_device_info()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Unsloth LLM Fine-Tuner for Ollama", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Unsloth LLM Fine-Tuner — Hugging Face Browser and Ollama Export")
        with gr.Tabs():
            with gr.Tab("Settings"):
                gr.Markdown("Configure tokens, endpoints and view system device info.")
                with gr.Row():
                    hf_token_in = gr.Textbox(label="Hugging Face Token", type="password", placeholder="hf_...", value="")
                    ollama_url_in = gr.Textbox(label="Ollama URL", value=settings.ollama_url)
                clear_token_cb = gr.Checkbox(label="Clear saved token", value=False)
                btn_save_settings = gr.Button("Save Settings")
                settings_status = gr.Textbox(label="Status", interactive=False)
                gr.Markdown("Current Settings (token not shown):")
                settings_view = gr.JSON(label="Settings")
                gr.Markdown("Device Info:")
                device_view = gr.JSON(label="Device")

            with gr.Tab("Models"):
                with gr.Row():
                    query = gr.Textbox(label="Query", value="llama3")
                    task = gr.Textbox(label="Task filter (e.g., text-generation)")
                    library = gr.Textbox(label="Library filter (e.g., transformers)")
                with gr.Row():
                    limit = gr.Number(label="Limit", value=20, precision=0)
                    sort = gr.Dropdown(["downloads", "likes", "lastModified"], value="downloads", label="Sort")
                    direction = gr.Dropdown(["desc", "asc"], value="desc", label="Direction")
                btn_search = gr.Button("Search Models")
                df_models = gr.Dataframe(headers=["id", "downloads", "likes", "library_name", "task"], interactive=False)
                with gr.Row():
                    repo_id = gr.Textbox(label="Model repo id (e.g., meta-llama/Meta-Llama-3-8B)")
                    btn_dl = gr.Button("Download")
                dl_status = gr.Textbox(label="Download status")
                with gr.Row():
                    btn_readme_m = gr.Button("Show README")
                md_model_readme = gr.Markdown()

            with gr.Tab("Datasets"):
                with gr.Row():
                    d_query = gr.Textbox(label="Query", value="alpaca")
                    d_task = gr.Textbox(label="Task filter (e.g., text2text-generation)")
                with gr.Row():
                    d_limit = gr.Number(label="Limit", value=20, precision=0)
                    d_sort = gr.Dropdown(["downloads", "likes", "lastModified"], value="downloads", label="Sort")
                    d_direction = gr.Dropdown(["desc", "asc"], value="desc", label="Direction")
                btn_dsearch = gr.Button("Search Datasets")
                df_datasets = gr.Dataframe(headers=["id", "downloads", "likes", "card_data"], interactive=False)
                with gr.Row():
                    dataset_id = gr.Textbox(label="Dataset id (e.g., yahma/alpaca-cleaned)")
                    btn_readme_d = gr.Button("Show README")
                md_dataset_readme = gr.Markdown()

            with gr.Tab("Data"):
                gr.Markdown("Upload files to data/ and browse.")
                up = gr.File(label="Upload files", file_count="multiple")
                btn_upload = gr.Button("Save to data/")
                up_status = gr.Textbox(label="Save status")
                btn_list_data = gr.Button("List data directory")
                df_data = gr.Dataframe(headers=["type", "path"], interactive=False)

            with gr.Tab("Outputs"):
                gr.Markdown("Browse outputs/")
                btn_list_outputs = gr.Button("List outputs directory")
                df_outputs = gr.Dataframe(headers=["type", "path"], interactive=False)

            with gr.Tab("Fine-tune"):
                gr.Markdown("Provide base model and dataset. This runs LoRA/QLoRA with Unsloth.")
                with gr.Row():
                    base_model_id = gr.Textbox(label="Base model id", value="unsloth/llama-3-8b-bnb-4bit")
                    dataset = gr.Textbox(label="Dataset (HF name or local path)", value="yahma/alpaca-cleaned")
                    text_field = gr.Textbox(label="Text field", value="text")
                with gr.Row():
                    out_dir = gr.Textbox(label="Output dir (optional)")
                    lora_r = gr.Number(label="LoRA r", value=16, precision=0)
                    lora_alpha = gr.Number(label="LoRA alpha", value=32, precision=0)
                    lora_dropout = gr.Number(label="LoRA dropout", value=0.05)
                with gr.Row():
                    learning_rate = gr.Number(label="Learning rate", value=2e-4)
                    per_device_train_batch_size = gr.Number(label="Batch size", value=1, precision=0)
                    gradient_accumulation_steps = gr.Number(label="Grad accum steps", value=8, precision=0)
                with gr.Row():
                    num_train_epochs = gr.Number(label="Epochs", value=1.0)
                    max_seq_length = gr.Number(label="Max seq len", value=2048, precision=0)
                    use_4bit = gr.Checkbox(label="Use 4-bit (QLoRA)", value=True)
                    bf16 = gr.Checkbox(label="bf16", value=False)
                btn_train = gr.Button("Start fine-tune")
                job_id = gr.Textbox(label="Job ID", interactive=False)
                status = gr.Textbox(label="Status")
                logs = gr.Textbox(label="Logs", lines=20)
                btn_refresh = gr.Button("Refresh status")

            with gr.Tab("Merge (LoRA→HF)"):
                gr.Markdown("Merge a LoRA adapter into the base model and save as a standard HF model.")
                with gr.Row():
                    merge_base = gr.Textbox(label="Base model id")
                    merge_adapter = gr.Textbox(label="Adapter dir (outputs/...) ")
                    merge_out_dir = gr.Textbox(label="Output dir (optional)")
                torch_dtype = gr.Dropdown(["auto", "float16", "bfloat16"], value="auto", label="torch dtype")
                btn_merge = gr.Button("Start Merge")
                merge_job_id = gr.Textbox(label="Job ID", interactive=False)
                merge_status = gr.Textbox(label="Status")
                merge_logs = gr.Textbox(label="Logs", lines=20)
                btn_merge_refresh = gr.Button("Refresh status")

            with gr.Tab("Inference"):
                gr.Markdown("Quick test generation with base or base+LoRA.")
                with gr.Row():
                    infer_model = gr.Textbox(label="Base model id", value="unsloth/llama-3-8b-bnb-4bit")
                    infer_adapter = gr.Textbox(label="Adapter dir (optional)")
                prompt = gr.Textbox(label="Prompt", value="Write a short poem about AI.", lines=4)
                with gr.Row():
                    infer_max_new = gr.Number(label="Max new tokens", value=128, precision=0)
                    infer_temp = gr.Number(label="Temperature", value=0.7)
                    infer_top_p = gr.Number(label="Top-p", value=0.9)
                    infer_use_4bit = gr.Checkbox(label="Use 4-bit", value=True)
                    infer_bf16 = gr.Checkbox(label="bf16", value=False)
                    infer_max_seq = gr.Number(label="Max seq len", value=4096, precision=0)
                btn_infer = gr.Button("Generate")
                infer_out = gr.Textbox(label="Output", lines=12)

            with gr.Tab("Jobs"):
                btn_list = gr.Button("Refresh")
                df_jobs = gr.Dataframe(headers=["id", "kind", "status", "start_time", "end_time"], interactive=False)

            with gr.Tab("Ollama Export"):
                gr.Markdown("Build a Modelfile and send to your Ollama server. Note: For LoRA adapters you may need to merge and convert to GGUF externally before running in Ollama.")
                with gr.Row():
                    ollama_url = gr.Textbox(label="Ollama URL", value=settings.ollama_url)
                    model_name = gr.Textbox(label="New model name", value="my-finetuned-model")
                base_from = gr.Textbox(label="FROM (base model or local gguf path)", value="llama3")
                template_text = gr.Textbox(label="TEMPLATE (optional)", value="{{ .Prompt }}")
                btn_create = gr.Button("Create/Update Model on Ollama")
                create_result = gr.JSON(label="Result")

        # Wiring
        btn_save_settings.click(_save_settings, [hf_token_in, ollama_url_in, clear_token_cb], [settings_status])
        btn_save_settings.click(_get_settings, None, [settings_view])
        # Populate settings and device info on load
        demo.load(_get_settings, None, [settings_view])
        demo.load(_device_info, None, [device_view])

        btn_search.click(_search_models, [query, limit, task, library, sort, direction], df_models)
        btn_dl.click(_download_model, [repo_id], [dl_status])
        btn_readme_m.click(_model_readme, [repo_id], [md_model_readme])
        btn_dsearch.click(_search_datasets, [d_query, d_limit, d_task, d_sort, d_direction], df_datasets)
        btn_readme_d.click(_dataset_readme, [dataset_id], [md_dataset_readme])

        btn_upload.click(_upload_to_data, [up], [up_status])
        btn_list_data.click(_list_data_dir, None, df_data)
        btn_list_outputs.click(_list_outputs_dir, None, df_outputs)

        btn_train.click(
            _start_finetune,
            [base_model_id, dataset, text_field, out_dir, lora_r, lora_alpha, lora_dropout, learning_rate,
             per_device_train_batch_size, gradient_accumulation_steps, num_train_epochs, max_seq_length, use_4bit, bf16],
            [job_id],
        )
        btn_refresh.click(_job_status, [job_id], [status, logs])

        btn_list.click(_list_jobs, None, df_jobs)

        btn_create.click(_create_ollama_model, [ollama_url, model_name, base_from, template_text], create_result)

        btn_merge.click(_start_merge, [merge_base, merge_adapter, merge_out_dir, torch_dtype], [merge_job_id])
        btn_merge_refresh.click(_job_status, [merge_job_id], [merge_status, merge_logs])

    return demo
