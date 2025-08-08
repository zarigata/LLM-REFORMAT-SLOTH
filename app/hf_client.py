from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

from huggingface_hub import HfApi, snapshot_download, ModelFilter, DatasetFilter, hf_hub_download

from .settings import settings


_api = HfApi()


def search_models(
    query: str = "",
    limit: int = 20,
    task: Optional[str] = None,
    library: Optional[str] = None,
    sort: Optional[str] = "downloads",
    direction: Optional[str] = "desc",
) -> List[Dict[str, Any]]:
    """Search Hugging Face models.
    Returns list of dicts with key metadata.
    """
    filt = ModelFilter(
        task=task,
        library=library,
    )
    results = _api.list_models(
        search=query or None,
        filter=filt,
        sort=sort,
        direction=direction,
        limit=limit,
    )
    rows = []
    for m in results:
        rows.append(
            {
                "id": m.id,
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "library_name": getattr(m, "library_name", None),
                "task": ",".join(m.pipeline_tag) if hasattr(m, "pipeline_tag") and m.pipeline_tag else getattr(m, "pipeline_tag", None),
            }
        )
    return rows


def search_datasets(
    query: str = "",
    limit: int = 20,
    task: Optional[str] = None,
    sort: Optional[str] = "downloads",
    direction: Optional[str] = "desc",
) -> List[Dict[str, Any]]:
    filt = DatasetFilter(task_categories=task)
    results = _api.list_datasets(
        search=query or None,
        filter=filt,
        sort=sort,
        direction=direction,
        limit=limit,
    )
    rows = []
    for d in results:
        rows.append(
            {
                "id": d.id,
                "downloads": getattr(d, "downloads", None),
                "likes": getattr(d, "likes", None),
                "card_data": getattr(d, "cardData", None),
            }
        )
    return rows


def download_model(repo_id: str, local_dir: Optional[str] = None) -> str:
    """Download a model snapshot to local_dir. Returns path.
    Requires HF token for some repos.
    """
    local_dir = local_dir or os.path.join(settings.hf_cache_dir, "models", repo_id.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=settings.hf_token,
        ignore_patterns=["*.msgpack", "*.h5"],
        revision=None,
        etag_timeout=120,
    )
    return local_dir


def _readme_path_from_info(info) -> Optional[str]:
    try:
        for sib in getattr(info, "siblings", []) or []:
            name = getattr(sib, "rfilename", "") or ""
            if name.lower() in {"readme.md", "readme", "readme.rst"}:
                return name
    except Exception:
        pass
    # default to README.md
    return "README.md"


def get_model_readme(repo_id: str) -> str:
    """Return README markdown for a model repo, or a message if not found."""
    try:
        info = _api.model_info(repo_id, token=settings.hf_token)
        fn = _readme_path_from_info(info)
        path = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="model", token=settings.hf_token)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Could not load README for {repo_id}: {e}"


def get_dataset_readme(repo_id: str) -> str:
    """Return README markdown for a dataset repo, or a message if not found."""
    try:
        info = _api.dataset_info(repo_id, token=settings.hf_token)
        fn = _readme_path_from_info(info)
        path = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset", token=settings.hf_token)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Could not load README for dataset {repo_id}: {e}"
