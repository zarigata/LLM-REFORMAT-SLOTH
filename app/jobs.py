from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import json
import os

from . import settings as app_settings


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued, running, completed, failed
    params: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        # Trim to last 1000 lines to avoid memory bloat
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
        _save_all()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "params": self.params,
            "logs": self.logs,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Job":
        return Job(
            id=d.get("id") or str(uuid.uuid4()),
            kind=d.get("kind") or "unknown",
            status=d.get("status") or "queued",
            params=d.get("params") or {},
            logs=d.get("logs") or [],
            start_time=d.get("start_time"),
            end_time=d.get("end_time"),
        )


_JOBS: Dict[str, Job] = {}
_LOCK = threading.Lock()
_JOBS_PATH = os.path.join(app_settings.settings.outputs_dir, "jobs.json")


def _save_all() -> None:
    try:
        os.makedirs(app_settings.settings.outputs_dir, exist_ok=True)
        with _LOCK:
            payload = [j.to_dict() for j in _JOBS.values()]
        tmp = _JOBS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, _JOBS_PATH)
    except Exception:
        # Best-effort persistence; avoid crashing on IO errors
        pass


def _load_all() -> None:
    try:
        if not os.path.exists(_JOBS_PATH):
            return
        with open(_JOBS_PATH, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            return
        with _LOCK:
            for d in arr:
                try:
                    j = Job.from_dict(d)
                    # If app crashed mid-run, mark as failed-stopped
                    if j.status == "running":
                        j.status = "failed"
                        j.logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Job marked failed after restart")
                        j.end_time = j.end_time or time.time()
                    _JOBS[j.id] = j
                except Exception:
                    continue
    except Exception:
        pass


def create_job(kind: str, params: Dict[str, Any]) -> Job:
    with _LOCK:
        jid = str(uuid.uuid4())
        job = Job(id=jid, kind=kind, params=params)
        _JOBS[jid] = job
        _save_all()
        return job


def get_job(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> List[Dict[str, Any]]:
    with _LOCK:
        return [
            {
                "id": j.id,
                "kind": j.kind,
                "status": j.status,
                "start_time": j.start_time,
                "end_time": j.end_time,
            }
            for j in _JOBS.values()
        ]


def run_in_thread(job: Job, target: Callable[[Job], None]) -> None:
    def _wrap():
        job.status = "running"
        job.start_time = time.time()
        _save_all()
        try:
            target(job)
            if job.status != "failed":
                job.status = "completed"
        except Exception as e:
            job.log(f"Exception: {e}")
            job.status = "failed"
        finally:
            job.end_time = time.time()
            _save_all()

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()


# Load persisted jobs on module import
_load_all()
