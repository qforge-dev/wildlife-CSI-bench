"""Persistent, suite-local cache for exact iNaturalist name resolutions."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from wildlife_csi.ports import TaxonResolver


def normalized_name(name: str) -> str:
    return " ".join(name.casefold().split())


class CachedTaxonResolver:
    """Share immutable name lookups across models and serialize cache misses.

    The run's JSONL resolution file remains the auditable scoring snapshot. This
    database only avoids repeating a lookup already made for the same name.
    """

    def __init__(
        self,
        upstream: TaxonResolver,
        path: str | Path,
        *,
        min_lookup_interval_s: float = 1.1,
    ):
        self.upstream = upstream
        self.path = Path(path)
        self.min_lookup_interval_s = min_lookup_interval_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS names (name TEXT PRIMARY KEY, resolution TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=900)

    def resolve_name(self, name: str) -> dict | None:
        key = normalized_name(name)
        with closing(self._connect()) as db, db:
            # Keep the lock through the API call: concurrent scorers cannot make
            # the same lookup twice or bypass the shared request interval.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT resolution FROM names WHERE name = ?", (key,)).fetchone()
            if row is not None:
                return json.loads(row[0])
            previous = db.execute(
                "SELECT value FROM metadata WHERE key = 'last_lookup_at'"
            ).fetchone()
            if previous:
                time.sleep(max(0.0, previous[0] + self.min_lookup_interval_s - time.time()))
            resolution = self.upstream.resolve_name(name)
            db.execute(
                "INSERT INTO names(name, resolution) VALUES (?, ?)",
                (key, json.dumps(resolution, sort_keys=True)),
            )
            db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_lookup_at', ?)",
                (time.time(),),
            )
            return resolution
