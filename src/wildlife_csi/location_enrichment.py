"""Resolve country context from public observation provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wildlife_csi.animalclue import REPOS
from wildlife_csi.source import INaturalist
from wildlife_csi.suite import validate_suite


def inspect_hf_info() -> dict[str, dict]:
    """Check pinned per-repository metadata before using iNaturalist."""
    from huggingface_hub import hf_hub_download

    result = {}
    for clue, spec in REPOS.items():
        path = Path(
            hf_hub_download(
                spec["repo"], spec["info"], repo_type="dataset", revision=spec["revision"]
            )
        )
        with path.open(newline="") as stream:
            columns = csv.DictReader(stream).fieldnames or []
        result[clue] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "info_file": spec["info"],
            "info_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns": columns,
        }
        if any(
            any(word in col.casefold() for word in ("location", "country", "latitude", "longitude"))
            for col in columns
        ):
            raise ValueError(
                f"HF metadata gained a location column for {clue}; inspect its row mapping"
            )
    return result


class JsonlCheckpoint:
    """Append-only local cache for public API fields needed to derive countries."""

    def __init__(self, path: str | Path, key: str):
        self.path = Path(path)
        self.key = key
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row[key]
                if value in self.rows:
                    raise ValueError(f"duplicate checkpoint {key}: {value}")
                self.rows[value] = row

    def put(self, row: dict) -> None:
        value = row[self.key]
        if value in self.rows:
            raise ValueError(f"checkpoint already contains {value}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows[value] = row


def fetch_observations(tasks: list[dict], api, checkpoint: JsonlCheckpoint) -> dict[str, dict]:
    ids = [task["source"]["observation_id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate observation ID")
    missing = [obs_id for obs_id in ids if obs_id not in checkpoint.rows]
    for start in range(0, len(missing), 10):
        chunk = missing[start : start + 10]
        found = api.observations(chunk, require_all=False)
        for obs_id in chunk:
            obs = found.get(obs_id)
            checkpoint.put(
                {
                    "observation_id": obs_id,
                    "found": obs is not None,
                    "place_ids": list(obs.get("place_ids") or []) if obs else [],
                    "place_guess": obs.get("place_guess") if obs else None,
                    "geoprivacy": obs.get("geoprivacy") if obs else None,
                    "taxon_id": int(obs.get("taxon", {}).get("id"))
                    if obs and obs.get("taxon", {}).get("id")
                    else None,
                }
            )
    return {obs_id: checkpoint.rows[obs_id] for obs_id in ids}


def fetch_places(
    observations: dict[str, dict], api, checkpoint: JsonlCheckpoint
) -> dict[int, dict]:
    ids = sorted({int(pid) for obs in observations.values() for pid in obs["place_ids"]})
    missing = [pid for pid in ids if pid not in checkpoint.rows]
    for start in range(0, len(missing), 50):
        chunk = missing[start : start + 50]
        found = api.places(chunk)
        for pid in chunk:
            place = found.get(pid)
            checkpoint.put(
                {
                    "place_id": pid,
                    "found": place is not None,
                    "name": place.get("name") if place else None,
                    "admin_level": place.get("admin_level") if place else None,
                    "place_type": place.get("place_type") if place else None,
                }
            )
    return {pid: checkpoint.rows[pid] for pid in ids}


def _country_place(place: dict) -> bool:
    # iNaturalist represents Taiwan as admin 0 with no place_type.
    return place["found"] and place["admin_level"] == 0 and (
        place["place_type"] == 12 or place["name"] == "Taiwan"
    )


def country_names(places: dict[int, dict]) -> dict[str, str]:
    return {
        place["name"].casefold(): place["name"]
        for place in places.values()
        if _country_place(place) and place["name"]
    }


def observation_country(
    observation: dict, places: dict[int, dict], known_country_names: dict[str, str] | None = None
) -> dict:
    countries = {
        int(pid): places[int(pid)]["name"]
        for pid in observation["place_ids"]
        if int(pid) in places
        and _country_place(places[int(pid)])
    }
    geoprivacy = observation.get("geoprivacy") or "open"
    guess_tail = (observation.get("place_guess") or "").rsplit(",", 1)[-1].strip().casefold()
    guess_country = (known_country_names or country_names(places)).get(guess_tail)
    if len(countries) == 1:
        place_id, name = next(iter(countries.items()))
        if guess_country is not None and guess_country != name:
            return {
                "country": name,
                "level": "country",
                "basis": "observation_public_place_with_text_conflict",
                "is_observation_location": True,
                "source": "iNaturalist",
                "source_place_id": place_id,
                "geoprivacy": geoprivacy,
                "candidate_countries": sorted((name, guess_country)),
            }
        return {
            "country": name,
            "level": "country",
            "basis": "observation_public_place",
            "is_observation_location": True,
            "source": "iNaturalist",
            "source_place_id": place_id,
            "geoprivacy": geoprivacy,
        }
    territories = {
        int(pid): places[int(pid)]["name"]
        for pid in observation["place_ids"]
        if int(pid) in places and places[int(pid)]["found"]
        and places[int(pid)]["admin_level"] == 0
        and places[int(pid)]["name"] == "Puerto Rico"
    }
    if len(territories) == 1:
        place_id = next(iter(territories))
        return {
            "country": "United States",
            "region": "Puerto Rico",
            "level": "country",
            "basis": "observation_public_territory",
            "is_observation_location": True,
            "source": "iNaturalist",
            "source_place_id": place_id,
            "geoprivacy": geoprivacy,
        }
    return {
        "country": None,
        "level": "country",
        "basis": "unresolved_observation_location",
        "is_observation_location": False,
        "source": "iNaturalist",
        "source_place_id": None,
        "geoprivacy": geoprivacy,
        "candidate_countries": sorted(countries.values()),
    }


def build_observation_locations(
    tasks_path: str | Path,
    out_dir: str | Path,
    cache_dir: str | Path,
    *,
    api=None,
    hf_info: dict | None = None,
    range_examples_path: str | Path | None = None,
) -> dict:
    """Write a review snapshot without changing the frozen suite."""
    tasks, suite = validate_suite(tasks_path)
    info = hf_info if hf_info is not None else inspect_hf_info()
    api = api if api is not None else INaturalist()
    cache_dir = Path(cache_dir)
    observations = fetch_observations(
        tasks, api, JsonlCheckpoint(cache_dir / "observations.jsonl", "observation_id")
    )
    places = fetch_places(
        observations, api, JsonlCheckpoint(cache_dir / "places.jsonl", "place_id")
    )
    known_country_names = country_names(places)
    range_examples = {}
    if range_examples_path is not None and Path(range_examples_path).exists():
        for line in Path(range_examples_path).read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            obs_id = entry["observation_id"]
            if obs_id in range_examples:
                raise ValueError(f"duplicate species range example for observation {obs_id}")
            if not entry["country"] or not entry["source_url"].startswith("https://"):
                raise ValueError(f"invalid species range example for observation {obs_id}")
            range_examples[obs_id] = entry
    rows, enriched = [], []
    counts = Counter()
    for task in tasks:
        obs_id = task["source"]["observation_id"]
        obs = observations[obs_id]
        location = observation_country(obs, places, known_country_names)
        if not obs["found"]:
            location["basis"] = "observation_unavailable"
        if location["country"] is None and obs_id in range_examples:
            example = range_examples[obs_id]
            if example["species"] != task["correct_taxon"]:
                raise ValueError(f"species range example does not match frozen truth: {obs_id}")
            location = {
                "country": example["country"],
                "level": "country",
                "basis": "species_occurrence_example",
                "is_observation_location": False,
                "source": example["source"],
                "source_url": example["source_url"],
                "source_place_id": None,
                "geoprivacy": obs.get("geoprivacy") or "open",
            }
        location["observation_id"] = obs_id
        rows.append({"task_id": task["task_id"], "location": location})
        enriched.append({**task, "location": location})
        counts[location["basis"]] += 1
        if location["country"]:
            counts["country_present"] += 1
        if location["geoprivacy"] != "open":
            counts[f"geoprivacy_{location['geoprivacy']}"] += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    location_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    draft_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched).encode()
    (out_dir / "locations.jsonl").write_bytes(location_bytes)
    (out_dir / "tasks-with-location.jsonl").write_bytes(draft_bytes)
    result = {
        "source_suite_tasks_hash": suite["tasks_hash"],
        "tasks": len(tasks),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_info": info,
        "counts": dict(counts),
        "locations_sha256": hashlib.sha256(location_bytes).hexdigest(),
        "draft_tasks_sha256": hashlib.sha256(draft_bytes).hexdigest(),
        "range_examples_sha256": hashlib.sha256(Path(range_examples_path).read_bytes()).hexdigest()
        if range_examples_path is not None and Path(range_examples_path).exists()
        else None,
        "note": "Review snapshot only; frozen tasks and prompts are unchanged.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
