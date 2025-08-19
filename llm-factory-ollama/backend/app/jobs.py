import os
import threading
import time
import uuid
from typing import Callable, Dict, Optional


class Job:
    def __init__(self, kind: str, payload: dict):
        self.id = str(uuid.uuid4())
        self.kind = kind
        self.payload = payload
        self.status = "queued"  # queued, running, done, error
        self.progress = 0
        self.logs = []
        self.error: Optional[str] = None
        self.artifacts: Dict[str, str] = {}
        self.started_at = None
        self.finished_at = None

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        # persist tail to disk
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", f"{self.id}.log"), "a", encoding="utf-8") as f:
            f.write(self.logs[-1] + "\n")


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create(self, kind: str, payload: dict, target: Callable[[Job], None]) -> str:
        job = Job(kind, payload)
        self.jobs[job.id] = job

        def runner():
            try:
                job.status = "running"
                job.started_at = time.time()
                target(job)
                job.status = "done"
            except Exception as e:
                job.error = str(e)
                job.status = "error"
                job.log(f"Error: {e}")
            finally:
                job.finished_at = time.time()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job.id

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)


manager = JobManager()
