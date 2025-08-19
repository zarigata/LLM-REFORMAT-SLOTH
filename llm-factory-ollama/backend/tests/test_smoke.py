import os
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_dry_run_pipeline():
    r = client.post("/api/fine-tune", json={"base_model_source": "huggingface", "target_gpu": "cpu", "dry_run": True})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # poll status
    for _ in range(30):
        s = client.get(f"/api/status/{job_id}").json()
        if s["status"] in ("done", "error"):
            break
    assert s["status"] == "done"
    model_id = s["artifacts"].get("model_id")
    assert model_id

    # export
    r2 = client.post(f"/api/export/{model_id}", json={"model_id": model_id, "export_format": "gguf"})
    assert r2.status_code == 200
