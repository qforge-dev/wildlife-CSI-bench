"""Official image preparation and immutable country-aware CSI suite manifests.

The image policy is frozen here: EXIF transpose, RGB, 320px minimum short
side, 1536px thumbnail, JPEG q90 with metadata stripped, 3MB cap.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

from wildlife_csi.animalclue import REPOS
from wildlife_csi.parse import SYSTEM_PROMPT, user_prompt
from wildlife_csi.source import photo_reference

PROTOCOL_VERSION = 1
SUITE = "wildlife-csi-country-v1"
SEED = 42
PER_TYPE = 400


def prepare_image(data: bytes) -> tuple[bytes, dict]:
    with Image.open(io.BytesIO(data)) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated_or_multiframe")
        source.load()
        oriented = ImageOps.exif_transpose(source)
        if min(oriented.size) < 320:
            raise ValueError("source_resolution_below_320")
        rgb = Image.new("RGB", oriented.size, "white")
        if "A" in oriented.getbands():
            rgb.paste(oriented, mask=oriented.getchannel("A"))
        else:
            rgb.paste(oriented.convert("RGB"))
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        source_size = list(rgb.size)
        rgb.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        if min(rgb.size) < 320:
            raise ValueError("aspect_ratio_too_extreme")
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        out = io.BytesIO()
        clean.save(out, "JPEG", quality=90, optimize=True)
        payload = out.getvalue()
        if len(payload) > 3_000_000:
            raise ValueError("normalized_image_exceeds_3MB")
        return payload, {
            "source_size": source_size,
            "width": clean.width,
            "height": clean.height,
            "pixel_sha256": pixels,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }


def tasks_hash(tasks: list[dict]) -> str:
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda r: r["task_id"]):
        h.update(json.dumps(t, sort_keys=True).encode())
    return h.hexdigest()


def build_task(clue: str, image_sha: str, source: dict, location: dict) -> dict:
    truth = source["inat"]
    return {
        "task_id": f"{SUITE}:{clue}:{image_sha[:16]}",
        "task_type": "animal-trace-open-id",
        "protocol_version": PROTOCOL_VERSION,
        "clue_type": clue,
        "image_sha256": image_sha,
        "correct_taxon": truth["species"],
        "correct_taxon_id": truth["species_id"],
        "correct_genus": truth["genus"],
        "correct_genus_id": truth["genus_id"],
        "correct_family": truth["family"],
        "correct_family_id": truth["family_id"],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt(location),
        "location": location,
        "source": source,
    }


def _validate_task(task: dict) -> None:
    if (
        task.get("protocol_version") != PROTOCOL_VERSION
        or task.get("system_prompt") != SYSTEM_PROMPT
        or task.get("user_prompt") != user_prompt(task["location"])
    ):
        raise ValueError("suite protocol or prompt mismatch")
    if task.get("clue_type") not in REPOS:
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
        or location.get("basis") not in {
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


def write_manifest(
    out_dir: str | Path,
    tasks: list[dict],
    seed: int,
    exclusions: list[dict] | None = None,
    replace: bool = False,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not replace:
        raise FileExistsError(f"frozen suite already exists: {out}; choose a new --out")
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("empty suite or duplicate task IDs")
    if len({t["source"]["observation_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate source observations")
    for task in tasks:
        _validate_task(task)
        image = out / "images" / f"{task['image_sha256']}.jpg"
        if (
            not image.exists()
            or hashlib.sha256(image.read_bytes()).hexdigest() != task["image_sha256"]
        ):
            raise ValueError(f"missing/corrupt image: {image}")
    sources = [{"task_id": t["task_id"], "source": t["source"]} for t in tasks]
    source_bytes = ("\n".join(json.dumps(s, sort_keys=True) for s in sources) + "\n").encode()
    locations = [{"task_id": t["task_id"], "location": t["location"]} for t in tasks]
    location_bytes = ("\n".join(json.dumps(s, sort_keys=True) for s in locations) + "\n").encode()
    manifest = {
        "suite": SUITE,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "tasks": len(tasks),
        "tasks_hash": tasks_hash(tasks),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "locations_sha256": hashlib.sha256(location_bytes).hexdigest(),
        "per_type": {c: sum(1 for t in tasks if t["clue_type"] == c) for c in REPOS},
    }
    (out / "tasks.jsonl").write_text("\n".join(json.dumps(t, sort_keys=True) for t in tasks) + "\n")
    (out / "sources.jsonl").write_bytes(source_bytes)
    (out / "locations.jsonl").write_bytes(location_bytes)
    (out / "exclusions.jsonl").write_text(
        "\n".join(json.dumps(e, sort_keys=True) for e in (exclusions or [])) + "\n"
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


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
        if storage.get("kind") != "s3" or not storage.get("private"):
            raise ValueError("unsupported or nonprivate suite storage")
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
