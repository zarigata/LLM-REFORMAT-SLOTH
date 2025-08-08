from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


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


_JOBS: Dict[str, Job] = {}
_LOCK = threading.Lock()


def create_job(kind: str, params: Dict[str, Any]) -> Job:
    with _LOCK:
        jid = str(uuid.uuid4())
        job = Job(id=jid, kind=kind, params=params)
        _JOBS[jid] = job
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
        try:
            target(job)
            if job.status != "failed":
                job.status = "completed"
        except Exception as e:
            job.log(f"Exception: {e}")
            job.status = "failed"
        finally:
            job.end_time = time.time()

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
