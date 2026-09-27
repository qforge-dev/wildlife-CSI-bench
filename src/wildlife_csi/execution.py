"""Provider-independent execution, retries, checkpoints, and run provenance."""

from __future__ import annotations

import concurrent.futures
import hashlib
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from wildlife_csi.ports import ImageStore, PredictionStore, VisionAdapter

TRANSIENT = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connecterror",
    "unexpected_eof",
    "eof occurred",
    "overloaded",
    "rate limit",
    "temporarily",
)
RECORD_SCHEMA = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_hash() -> str:
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class RunSettings:
    max_workers: int = 2
    max_cost: float | None = None
    retries: int = 3
    backoff_s: float = 1.0

    def __post_init__(self):
        if self.max_workers < 1 or self.retries < 0 or self.backoff_s < 0:
            raise ValueError("invalid run settings")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must be nonnegative")


def _status(predictions: list[dict], info: dict) -> str:
    if info.get("finish_reason") in ("length", "max_tokens", "model_context_window_exceeded"):
        return "truncated"
    if info.get("finish_reason") in ("content_filter", "refusal"):
        return "refused"
    if not predictions or not str(predictions[0].get("taxon") or "").strip():
        return "empty"
    return "answered"


class RunEngine:
    """Orchestrate a frozen suite through injected adapter and storage ports."""

    def __init__(
        self,
        adapter: VisionAdapter,
        images: ImageStore,
        store: PredictionStore,
        *,
        now: Callable[[], str] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.adapter = adapter
        self.images = images
        self.store = store
        self.now = now
        self.monotonic = monotonic
        self.sleep = sleep

    def _call(self, task: dict, task_hash: str, settings: RunSettings) -> dict:
        image = self.images.load(task)
        context = {
            "system_prompt": task.get("system_prompt"),
            "user_prompt": task["user_prompt"],
            "task_id": task["task_id"],
        }
        task_start = self.monotonic()
        started_at = self.now()
        attempts = []
        for attempt in range(1, settings.retries + 2):
            attempt_start = self.monotonic()
            attempt_utc = self.now()
            try:
                predictions, info = self.adapter.predict(image, context)
                attempts.append(
                    {
                        "attempt": attempt,
                        "started_at_utc": attempt_utc,
                        "ended_at_utc": self.now(),
                        "duration_s": round(self.monotonic() - attempt_start, 6),
                        "outcome": "response",
                    }
                )
                return {
                    "capture_schema": RECORD_SCHEMA,
                    "task_id": task["task_id"],
                    "model_id": self.adapter.model_id,
                    "tasks_hash": task_hash,
                    "image_sha256": task["image_sha256"],
                    "input": {
                        "system_prompt": task.get("system_prompt"),
                        "prompt": task["user_prompt"],
                        "image_sha256": task["image_sha256"],
                        "image_bytes": len(image),
                    },
                    "predictions": predictions,
                    "status": _status(predictions, info),
                    "error": None,
                    "attempts": attempt,
                    "attempts_detail": attempts,
                    "started_at_utc": started_at,
                    "ended_at_utc": self.now(),
                    "latency_s": round(self.monotonic() - task_start, 6),
                    "finish_reason": info.get("finish_reason"),
                    "usage": info.get("usage", {}),
                    "estimated_cost_usd": info.get("estimated_cost_usd"),
                    "provider": info,
                }
            except Exception as exc:  # each failure is recorded and may be retried
                transient = any(
                    marker in f"{type(exc).__name__} {exc}".lower() for marker in TRANSIENT
                )
                record = {
                    "attempt": attempt,
                    "started_at_utc": attempt_utc,
                    "ended_at_utc": self.now(),
                    "duration_s": round(self.monotonic() - attempt_start, 6),
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "transient": transient,
                }
                if hasattr(exc, "details"):
                    record["provider"] = exc.details
                retry = transient and attempt <= settings.retries
                if retry:
                    record["backoff_s"] = min(8.0, settings.backoff_s * 2 ** (attempt - 1))
                attempts.append(record)
                if retry:
                    self.sleep(record["backoff_s"])
                    continue
                return {
                    "capture_schema": RECORD_SCHEMA,
                    "task_id": task["task_id"],
                    "model_id": self.adapter.model_id,
                    "tasks_hash": task_hash,
                    "image_sha256": task["image_sha256"],
                    "input": {
                        "system_prompt": task.get("system_prompt"),
                        "prompt": task["user_prompt"],
                        "image_sha256": task["image_sha256"],
                        "image_bytes": len(image),
                    },
                    "predictions": [],
                    "status": "transport_error",
                    "error": str(exc),
                    "attempts": attempt,
                    "attempts_detail": attempts,
                    "started_at_utc": started_at,
                    "ended_at_utc": self.now(),
                    "latency_s": round(self.monotonic() - task_start, 6),
                    "usage": {},
                    "provider": getattr(exc, "details", {}),
                }
        raise AssertionError("unreachable")

    def run(
        self, tasks: list[dict], suite_manifest: dict, settings: RunSettings = RunSettings()
    ) -> dict[str, Any]:
        task_hash = suite_manifest["tasks_hash"]
        expected = {task["task_id"]: task for task in tasks}
        if len(expected) != len(tasks):
            raise ValueError("duplicate task IDs")
        existing = self.store.read()
        run_manifest = self.store.load_manifest()
        if existing and run_manifest is None:
            raise ValueError("existing predictions lack a run manifest; start a new output path")
        config = self.adapter.public_config()
        if run_manifest is None:
            run_manifest = {
                "capture_schema": RECORD_SCHEMA,
                "suite": suite_manifest,
                "model_id": self.adapter.model_id,
                "adapter_config": config,
                "created_at_utc": self.now(),
                "python_version": platform.python_version(),
                "sessions": [],
            }
        elif (
            run_manifest.get("capture_schema") != RECORD_SCHEMA
            or run_manifest.get("suite", {}).get("tasks_hash") != task_hash
            or run_manifest.get("model_id") != self.adapter.model_id
            or run_manifest.get("adapter_config") != config
        ):
            raise ValueError("run manifest belongs to another suite/model/configuration")
        done = {}
        for row in existing:
            tid = row["task_id"]
            if (
                tid not in expected
                or tid in done
                or row.get("capture_schema") != RECORD_SCHEMA
                or row.get("tasks_hash") != task_hash
                or row.get("model_id") != self.adapter.model_id
                or row.get("image_sha256") != expected[tid]["image_sha256"]
            ):
                raise ValueError("existing prediction does not match this run")
            done[tid] = row
        self.adapter.seed_usage(existing)
        pending = [task for task in tasks if task["task_id"] not in done]
        session = {
            "started_at_utc": self.now(),
            "code_sha256": code_hash(),
            "settings": {
                "max_workers": settings.max_workers,
                "max_cost": settings.max_cost,
                "retries": settings.retries,
                "backoff_s": settings.backoff_s,
            },
            "already_done": len(done),
        }
        run_manifest["sessions"].append(session)
        self.store.save_manifest(run_manifest)
        written = 0
        stopped = False

        def cost_reached() -> bool:
            return (
                settings.max_cost is not None
                and self.adapter.estimated_cost() >= settings.max_cost
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
            running = {}
            next_index = 0
            while next_index < len(pending) or running:
                while (
                    next_index < len(pending)
                    and len(running) < settings.max_workers
                    and not cost_reached()
                ):
                    task = pending[next_index]
                    next_index += 1
                    running[pool.submit(self._call, task, task_hash, settings)] = task
                if not running:
                    stopped = next_index < len(pending)
                    break
                complete, _ = concurrent.futures.wait(
                    running, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in complete:
                    running.pop(future)
                    self.store.append(future.result())
                    written += 1
        session.update(
            {
                "ended_at_utc": self.now(),
                "written": written,
                "stopped_at_max_cost": stopped,
                "estimated_cost_usd": self.adapter.estimated_cost(),
            }
        )
        self.store.save_manifest(run_manifest)
        return {
            "model_id": self.adapter.model_id,
            "tasks_hash": task_hash,
            "total": len(tasks),
            "already_done": len(done),
            "written": written,
            "stopped_at_max_cost": stopped,
            "estimated_cost_usd": session["estimated_cost_usd"],
        }
