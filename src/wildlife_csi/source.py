"""iNaturalist name resolution and source-photo references."""

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
