"""Frozen iNaturalist observation taxonomy for AnimalClue image filenames.

The YOLO class number locates a trace; info_*.csv is not an ordered class map.
Species truth comes from the observation ID embedded in the source filename.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from urllib.parse import quote

import httpx

API = "https://api.inaturalist.org/v1"


def photo_reference(path: str) -> tuple[str, int]:
    match = re.match(r"^(\d+)_(\d+)(?:_|\.|$)", Path(path).name)
    if not match:
        raise ValueError(f"source image has no observation/photo index: {path}")
    return match.group(1), int(match.group(2))


class INaturalist:
    def __init__(self, client: httpx.Client | None = None, request_interval_s: float = 1.1):
        if request_interval_s < 0:
            raise ValueError("request interval must be nonnegative")
        self.client = client or httpx.Client(timeout=30, headers={"User-Agent": "wildlife-csi/0.1"})
        self._taxa: dict[int, dict] = {}
        self._request_interval_s = request_interval_s
        self._next_request_at = 0.0
        self._request_lock = threading.Lock()

    def _pace_request(self) -> None:
        with self._request_lock:
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_request_at = time.monotonic() + self._request_interval_s

    def _cooldown(self, seconds: float) -> None:
        with self._request_lock:
            self._next_request_at = max(self._next_request_at, time.monotonic() + seconds)

    def _get(self, path: str) -> list[dict]:
        for attempt in range(8):
            self._pace_request()
            try:
                response = self.client.get(f"{API}/{path}")
                response.raise_for_status()
                return response.json()["results"]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < 7:
                    try:
                        retry_after = float(exc.response.headers.get("Retry-After", "0"))
                    except ValueError:
                        retry_after = 0.0
                    self._cooldown(max(60.0, retry_after))
                    continue
                if attempt >= 3 or exc.response.status_code not in (500, 502, 503, 504):
                    raise
            except httpx.RequestError:
                if attempt >= 3:
                    raise
            if attempt < 3:
                time.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")

    def observations(self, ids: list[str], require_all: bool = True) -> dict[str, dict]:
        result = {}
        for offset in range(0, len(ids), 10):
            chunk = ids[offset : offset + 10]
            for row in self._get("observations/" + ",".join(chunk)):
                result[str(row["id"])] = row
        missing = set(ids) - result.keys()
        if missing and require_all:
            raise ValueError(f"missing iNaturalist observations: {sorted(missing)}")
        return result

    def taxa(self, ids: list[int]) -> dict[int, dict]:
        need = sorted(set(ids) - self._taxa.keys())
        for offset in range(0, len(need), 10):
            chunk = need[offset : offset + 10]
            for row in self._get("taxa/" + ",".join(map(str, chunk))):
                self._taxa[int(row["id"])] = row
        missing = set(ids) - self._taxa.keys()
        if missing:
            raise ValueError(f"missing iNaturalist taxa: {sorted(missing)}")
        return {i: self._taxa[i] for i in ids}

    def places(self, ids: list[int]) -> dict[int, dict]:
        """Look up public place records, including their administrative level."""
        result = {}
        for offset in range(0, len(ids), 50):
            chunk = ids[offset : offset + 50]
            for row in self._get("places/" + ",".join(map(str, chunk))):
                result[int(row["id"])] = row
        return result

    def resolve_name(self, name: str) -> dict | None:
        """Resolve an exact scientific/common name; ambiguous names remain unresolved."""
        folded = " ".join(name.casefold().split())
        if not folded or folded == "unknown":
            return None
        candidates = self._get("taxa/autocomplete?q=" + quote(name, safe=""))

        def matching(field: str) -> list[dict]:
            return [
                c
                for c in candidates
                if " ".join(str(c.get(field, "")).casefold().split()) == folded
            ]

        chosen = None
        for field in ("name", "preferred_common_name", "matched_term"):
            exact = matching(field)
            if not exact:
                continue
            if len(exact) == 1:
                chosen = exact[0]
            else:
                species = [c for c in exact if c.get("rank") == "species"]
                if len(species) == 1 and all(
                    c is species[0]
                    or species[0]["id"] in (c.get("ancestor_ids") or [])
                    or c["id"] in (species[0].get("ancestor_ids") or [])
                    for c in exact
                ):
                    chosen = species[0]
            break
        if chosen is None:
            return None
        taxon = self.taxa([int(chosen["id"])])[int(chosen["id"])]
        lineage = (taxon.get("ancestors") or []) + [taxon]

        def ancestor(rank: str) -> dict | None:
            return next((x for x in reversed(lineage) if x.get("rank") == rank), None)

        return {
            "taxon_id": int(taxon["id"]),
            "taxon": taxon["name"],
            "rank": taxon["rank"],
            **{
                f"{rank}_id": int(a["id"]) if (a := ancestor(rank)) else None
                for rank in ("species", "genus", "family")
            },
        }


def source_record(observation: dict, taxon: dict, image_path: str) -> dict:
    obs_id, photo_index = photo_reference(image_path)
    if str(observation.get("id")) != obs_id:
        raise ValueError(f"observation ID mismatch for {image_path}")
    if observation.get("quality_grade") != "research":
        raise ValueError(f"observation {obs_id} is not research grade")
    if int(observation.get("taxon", {}).get("id", -1)) != int(taxon["id"]):
        raise ValueError(f"taxon ID mismatch for observation {obs_id}")
    if taxon.get("rank") not in ("species", "subspecies", "variety"):
        raise ValueError(f"observation {obs_id} is not identified to species")
    photos = observation.get("photos") or []
    if photo_index >= len(photos):
        raise ValueError(f"photo index {photo_index} missing from observation {obs_id}")
    ancestors = taxon.get("ancestors") or []
    lineage = ancestors + [taxon]

    def rank(which: str) -> dict:
        candidates = [a for a in lineage if a.get("rank") == which]
        if not candidates:
            raise ValueError(f"observation {obs_id} has no {which} ancestor")
        return candidates[-1]

    species, genus, family = (rank(r) for r in ("species", "genus", "family"))
    photo = photos[photo_index]
    return {
        "observation_id": obs_id,
        "observation_url": f"https://www.inaturalist.org/observations/{obs_id}",
        "quality_grade": observation["quality_grade"],
        "observed_taxon_id": int(taxon["id"]),
        "observed_taxon": taxon["name"],
        "observed_rank": taxon["rank"],
        "species_id": int(species["id"]),
        "species": species["name"],
        "genus_id": int(genus["id"]),
        "genus": genus["name"],
        "family_id": int(family["id"]),
        "family": family["name"],
        "photo_index": photo_index,
        "photo_id": int(photo["id"]),
        "photo_license": photo.get("license_code"),
        "photo_attribution": photo.get("attribution"),
    }
