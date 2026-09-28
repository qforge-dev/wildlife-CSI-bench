"""Validate the frozen tasks, prompts, provenance, and image checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wildlife_csi.parse import SYSTEM_PROMPT, user_prompt
from wildlife_csi.source import photo_reference

PROTOCOL_VERSION = 1
SUITE = "wildlife-csi-country-v1"
CLUE_TYPES = ("bone", "egg", "feather", "feces", "footprint")


def tasks_hash(tasks: list[dict]) -> str:
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda r: r["task_id"]):
        h.update(json.dumps(t, sort_keys=True).encode())
    return h.hexdigest()


def _validate_task(task: dict) -> None:
    if (
        task.get("protocol_version") != PROTOCOL_VERSION
        or task.get("system_prompt") != SYSTEM_PROMPT
        or task.get("user_prompt") != user_prompt(task["location"])
    ):
        raise ValueError("suite protocol or prompt mismatch")
    if task.get("clue_type") not in CLUE_TYPES:
        raise ValueError("unknown clue type")
    sha = task.get("image_sha256", "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid image SHA-256")
    if task.get("task_id") != f"{SUITE}:{task['clue_type']}:{sha[:16]}":
        raise ValueError("task ID does not match suite/image")
    location = task["location"]
    if (
        location.get("level") != "country"
        or not isinstance(location.get("country"), str)
        or not location["country"].strip()
        or location.get("observation_id") != task["source"].get("observation_id")
        or location.get("basis")
        not in {
            "observation_public_place",
            "observation_public_place_with_text_conflict",
            "observation_public_territory",
            "species_occurrence_example",
        }
    ):
        raise ValueError("invalid task location")
    is_observed = location["basis"] != "species_occurrence_example"
    if location.get("is_observation_location") is not is_observed:
        raise ValueError("location provenance flag does not match its basis")
    if not is_observed and not str(location.get("source_url", "")).startswith("https://"):
        raise ValueError("range example must cite a source URL")
    source = task["source"]
    truth = source["inat"]
    obs, index = photo_reference(source["image_path"])
    if (
        source.get("observation_id") != obs
        or truth.get("observation_id") != obs
        or truth.get("photo_index") != index
    ):
        raise ValueError("source observation/photo reference mismatch")
    for field, source_field in (
        ("correct_taxon", "species"),
        ("correct_taxon_id", "species_id"),
        ("correct_genus", "genus"),
        ("correct_genus_id", "genus_id"),
        ("correct_family", "family"),
        ("correct_family_id", "family_id"),
    ):
        if task.get(field) != truth.get(source_field):
            raise ValueError(f"{field} differs from source observation")


def validate_suite(
    tasks_path: str | Path, images_dir: str | Path | None = None
) -> tuple[list[dict], dict]:
    path = Path(tasks_path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if manifest.get("suite") != SUITE or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("suite or protocol version mismatch")
    if manifest.get("tasks") != len(tasks) or manifest.get("tasks_hash") != tasks_hash(tasks):
        raise ValueError("manifest task hash/count mismatch")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs")
    if len({t["source"]["observation_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate source observations")
    source_bytes = (path.parent / "sources.jsonl").read_bytes()
    if manifest.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("source snapshot hash mismatch")
    sources = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if sources != [{"task_id": t["task_id"], "source": t["source"]} for t in tasks]:
        raise ValueError("source snapshot differs from tasks")
    location_bytes = (path.parent / "locations.jsonl").read_bytes()
    if manifest.get("locations_sha256") != hashlib.sha256(location_bytes).hexdigest():
        raise ValueError("location snapshot hash mismatch")
    locations = [json.loads(line) for line in location_bytes.splitlines() if line.strip()]
    if locations != [{"task_id": t["task_id"], "location": t["location"]} for t in tasks]:
        raise ValueError("location snapshot differs from tasks")
    if selection := manifest.get("selection"):
        ledger_bytes = (path.parent / "selection.jsonl").read_bytes()
        if hashlib.sha256(ledger_bytes).hexdigest() != selection.get("ledger_sha256"):
            raise ValueError("selection ledger hash mismatch")
        ledger = [json.loads(line) for line in ledger_bytes.splitlines() if line.strip()]
        chosen = {row["task_id"] for row in ledger if row["selected"]}
        if chosen != {task["task_id"] for task in tasks}:
            raise ValueError("selection ledger differs from tasks")
    storage = manifest.get("storage")
    if storage:
        if storage.get("kind") != "s3":
            raise ValueError("unsupported suite storage")
        bucket = storage["bucket"]
        prefix = storage["image_prefix"]
        for task in tasks:
            sha = task["image_sha256"]
            if task.get("image_s3_uri") != f"s3://{bucket}/{prefix}{sha[:2]}/{sha}.jpg":
                raise ValueError("task image S3 URI does not match suite storage")
    for task in tasks:
        _validate_task(task)
        if task["protocol_version"] != manifest["protocol_version"]:
            raise ValueError("task protocol differs from manifest")
        if images_dir is not None:
            image = Path(images_dir) / f"{task['image_sha256']}.jpg"
            if (
                not image.exists()
                or hashlib.sha256(image.read_bytes()).hexdigest() != task["image_sha256"]
            ):
                raise ValueError(f"missing/corrupt image: {image}")
    return tasks, manifest
