"""Small synthetic suites for runner and scorer tests."""

import hashlib
import json
from pathlib import Path

from wildlife_csi.parse import SYSTEM_PROMPT, user_prompt
from wildlife_csi.suite import CLUE_TYPES, PROTOCOL_VERSION, SUITE, _validate_task, tasks_hash


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
        "per_type": {c: sum(1 for t in tasks if t["clue_type"] == c) for c in CLUE_TYPES},
    }
    (out / "tasks.jsonl").write_text("\n".join(json.dumps(t, sort_keys=True) for t in tasks) + "\n")
    (out / "sources.jsonl").write_bytes(source_bytes)
    (out / "locations.jsonl").write_bytes(location_bytes)
    (out / "exclusions.jsonl").write_text(
        "\n".join(json.dumps(e, sort_keys=True) for e in (exclusions or [])) + "\n"
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
