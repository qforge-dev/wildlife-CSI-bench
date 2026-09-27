"""Filesystem stores for prepared images, run records, and taxon resolutions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class LocalImageStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load(self, task: dict[str, Any]) -> bytes:
        image = self.directory / f"{task['image_sha256']}.jpg"
        payload = image.read_bytes()
        if hashlib.sha256(payload).hexdigest() != task["image_sha256"]:
            raise ValueError(f"image hash mismatch: {image}")
        return payload


class JsonlPredictionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest_path = self.path.with_name("run_manifest.json")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text())

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.manifest_path)


class JsonlResolutionStore:
    def __init__(self, path: str | Path, predictions_sha256: str, tasks_hash: str):
        self.path = Path(path)
        self.predictions_sha256 = predictions_sha256
        self.tasks_hash = tasks_hash
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    row.get("predictions_sha256") != predictions_sha256
                    or row.get("tasks_hash") != tasks_hash
                ):
                    raise ValueError("resolution cache belongs to another prediction/task snapshot")
                if row["task_id"] in self.entries:
                    raise ValueError("duplicate resolution cache task ID")
                self.entries[row["task_id"]] = row

    def get(self, task_id: str) -> dict | None:
        return self.entries.get(task_id)

    def put(self, entry: dict) -> None:
        row = {
            **entry,
            "predictions_sha256": self.predictions_sha256,
            "tasks_hash": self.tasks_hash,
        }
        if row["task_id"] in self.entries:
            raise ValueError("resolution already recorded")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.entries[row["task_id"]] = row


class JsonlExtractionStore:
    """Append-only extractor calls, bound to the input snapshot and extractor prompt/model."""

    def __init__(
        self, path: str | Path, predictions_sha256: str, tasks_hash: str, extractor_identity: str
    ):
        self.path = Path(path)
        self.binding = {
            "predictions_sha256": predictions_sha256,
            "tasks_hash": tasks_hash,
            "extractor_identity": extractor_identity,
        }
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if any(row.get(key) != value for key, value in self.binding.items()):
                    raise ValueError(
                        "extractor cache belongs to another prediction/task/extractor snapshot"
                    )
                previous = self.entries.get(row["task_id"])
                if previous and previous["extraction"].get("status") != "provider_error":
                    raise ValueError("duplicate completed extractor cache task ID")
                self.entries[row["task_id"]] = row

    def get(self, task_id: str) -> dict | None:
        return self.entries.get(task_id)

    def put(self, entry: dict) -> None:
        row = {**entry, **self.binding}
        previous = self.entries.get(row["task_id"])
        if previous and previous["extraction"].get("status") != "provider_error":
            raise ValueError("extractor already recorded")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.entries[row["task_id"]] = row
