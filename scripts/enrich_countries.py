"""Recheck public observation countries without modifying the frozen suite."""

import json
from pathlib import Path

from wildlife_csi.location_enrichment import build_observation_locations


if __name__ == "__main__":
    root = Path("data/benchmarks/wildlife-csi-country-v1-2000")
    result = build_observation_locations(
        root / "tasks.jsonl",
        Path("data/work/csi-country-refresh"),
        Path("data/work/csi-country-cache"),
        range_examples_path=root / "species-range-examples.jsonl",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
