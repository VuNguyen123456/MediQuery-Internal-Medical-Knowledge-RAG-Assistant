"""
job_store.py — File-based upload job status (shared via documents PVC on K8s).

Stages: parsing → chunking → embedding → upserting → done | error
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

STAGES = ("parsing", "chunking", "embedding", "upserting", "done")


def jobs_dir(docs_dir: Path) -> Path:
    directory = docs_dir / ".upload-jobs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _job_path(docs_dir: Path, job_id: str) -> Path:
    return jobs_dir(docs_dir) / f"{job_id}.json"


def _write(docs_dir: Path, job_id: str, data: dict) -> None:
    path = _job_path(docs_dir, job_id)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_job(docs_dir: Path, filename: str, *, reindexed: bool) -> str:
    job_id = str(uuid.uuid4())
    _write(
        docs_dir,
        job_id,
        {
            "job_id": job_id,
            "filename": filename,
            "reindexed": reindexed,
            "status": "running",
            "stage": "parsing",
            "message": "Starting ingestion...",
            "progress": None,
            "result": None,
            "error": None,
        },
    )
    return job_id


def update_job(
    docs_dir: Path,
    job_id: str,
    stage: str,
    message: str | None = None,
    *,
    progress: dict | None = None,
) -> None:
    path = _job_path(docs_dir, job_id)
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "running"
    data["stage"] = stage
    if message is not None:
        data["message"] = message
    if progress is not None:
        data["progress"] = progress
    _write(docs_dir, job_id, data)


def complete_job(docs_dir: Path, job_id: str, result: dict) -> None:
    path = _job_path(docs_dir, job_id)
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "done"
    data["stage"] = "done"
    data["message"] = "Indexing complete"
    data["progress"] = None
    data["result"] = result
    data["error"] = None
    _write(docs_dir, job_id, data)


def fail_job(docs_dir: Path, job_id: str, error: str, stage: str | None = None) -> None:
    path = _job_path(docs_dir, job_id)
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "error"
    if stage:
        data["stage"] = stage
    data["message"] = error
    data["error"] = error
    data["progress"] = None
    _write(docs_dir, job_id, data)


def get_job(docs_dir: Path, job_id: str) -> dict | None:
    path = _job_path(docs_dir, job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_progress_callback(docs_dir: Path, job_id: str) -> Callable:
    """Return an on_progress handler for ingest_pdf."""

    def on_progress(
        stage: str,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        progress = None
        if current is not None and total is not None and total > 0:
            progress = {"current": current, "total": total}
        update_job(docs_dir, job_id, stage, message, progress=progress)

    return on_progress
