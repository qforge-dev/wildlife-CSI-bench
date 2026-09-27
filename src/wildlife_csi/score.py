"""Application wiring and report files for taxonomic scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wildlife_csi.execution import code_hash
from wildlife_csi.answer_extractor import load_extractor
from wildlife_csi.suite import validate_suite
from wildlife_csi.scoring import ScoreEngine, align_predictions
from wildlife_csi.source import INaturalist
from wildlife_csi.storage import JsonlExtractionStore, JsonlResolutionStore
from wildlife_csi.taxonomy_cache import CachedTaxonResolver


def score_run(
    tasks_path: str | Path,
    predictions_path: str | Path,
    *,
    offline: bool = False,
    extractor_config: str | Path = "configs/extractors/answer.yaml",
    output_dir: str | Path | None = None,
    resolver=None,
    resolution_store=None,
    extractor=None,
    extractor_store=None,
) -> dict:
    tasks, manifest = validate_suite(tasks_path)
    pred_path = Path(predictions_path)
    raw_bytes = pred_path.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    predictions, model_id = align_predictions(tasks, rows, manifest["tasks_hash"])
    out = Path(output_dir) if output_dir else pred_path.parent / "score"
    out.mkdir(parents=True, exist_ok=True)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    new_cache_path = out / "resolutions.jsonl"
    cache = (
        resolution_store
        if resolution_store is not None
        else JsonlResolutionStore(new_cache_path, raw_sha, manifest["tasks_hash"])
    )
    active_extractor = extractor if extractor is not None else load_extractor(extractor_config)
    active_extractor_store = (
        extractor_store
        if extractor_store is not None
        else JsonlExtractionStore(
            out / "extractions.jsonl", raw_sha, manifest["tasks_hash"], active_extractor.identity
        )
    )
    active_resolver = (
        resolver
        if resolver is not None
        else CachedTaxonResolver(
            INaturalist(), Path(tasks_path).parent / "runs/taxon-cache.sqlite3"
        )
    )
    summary, details = ScoreEngine(
        active_resolver,
        cache,
        active_extractor,
        active_extractor_store,
    ).score(tasks, predictions, offline=offline)
    result = {
        "suite": manifest["suite"],
        "tasks_hash": manifest["tasks_hash"],
        "source_sha256": manifest["source_sha256"],
        "predictions_sha256": raw_sha,
        "resolutions_sha256": (
            hashlib.sha256(cache.path.read_bytes()).hexdigest()
            if isinstance(cache, JsonlResolutionStore) and cache.path.exists()
            else None
        ),
        "extractions_sha256": (
            hashlib.sha256(active_extractor_store.path.read_bytes()).hexdigest()
            if isinstance(active_extractor_store, JsonlExtractionStore)
            and active_extractor_store.path.exists()
            else None
        ),
        "extractor_config": active_extractor.public_config,
        "extractor_identity": active_extractor.identity,
        "model_id": model_id,
        "code_sha256": code_hash(),
        **summary,
    }
    (out / "scores.jsonl").write_text(
        "\n".join(json.dumps(d, sort_keys=True) for d in details) + "\n"
    )
    (out / "review_queue.jsonl").write_text(
        "".join(json.dumps(d, sort_keys=True) + "\n" for d in details if d["review_required"])
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
