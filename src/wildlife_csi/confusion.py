"""Reproducible sparse species confusion matrices from recorded per-photo scores."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _species_key(taxon_id: int) -> str:
    return f"species:{taxon_id}"


def build_confusion_matrix(details: list[dict]) -> dict:
    """Count every scored photo, retaining higher taxa and non-answer outcomes."""
    if not details:
        raise ValueError("cannot build a confusion matrix without scores")
    if len({row["task_id"] for row in details}) != len(details):
        raise ValueError("duplicate task IDs in scores")

    truth_names: dict[int, str] = {}
    species_names: dict[int, set[str]] = defaultdict(set)
    observed_names: dict[int, set[str]] = defaultdict(set)
    for row in details:
        taxon_id = row["truth_species_id"]
        if taxon_id in truth_names and truth_names[taxon_id] != row["truth"]:
            raise ValueError(f"conflicting ground-truth names for species {taxon_id}")
        truth_names[taxon_id] = row["truth"]
        resolution = row.get("resolution") or {}
        species_id = resolution.get("species_id")
        name = resolution.get("taxon")
        if species_id is not None and name:
            observed_names[species_id].add(name)
            if resolution.get("rank") == "species":
                species_names[species_id].add(name)

    labels: dict[str, dict] = {}

    def species_label(taxon_id: int) -> str:
        key = _species_key(taxon_id)
        if key not in labels:
            names = sorted(species_names[taxon_id])
            labels[key] = {
                "kind": "species",
                "taxon_id": taxon_id,
                "rank": "species",
                "name": truth_names.get(taxon_id) or (names[0] if names else None),
                "observed_names": sorted(observed_names[taxon_id]),
            }
        return key

    def prediction_label(row: dict) -> str:
        level = row["level"]
        resolution = row.get("resolution") or {}
        species_id = resolution.get("species_id")
        if level == "exact":
            if species_id is not None and species_id != row["truth_species_id"]:
                raise ValueError("exact score conflicts with resolved species")
            # Canonical-name matches need no taxonomy lookup and can have no resolution.
            return species_label(row["truth_species_id"])
        if level in {"wrong", "genus_only", "family_only"}:
            if species_id is not None:
                if species_id == row["truth_species_id"]:
                    raise ValueError("non-exact score resolves to the true species")
                return species_label(species_id)
            if resolution.get("taxon_id") is None:
                raise ValueError("resolved score is missing its taxon ID")
            key = f"taxon:{resolution['taxon_id']}"
            labels[key] = {
                "kind": "higher_taxon",
                "taxon_id": resolution["taxon_id"],
                "rank": resolution["rank"],
                "name": resolution["taxon"],
            }
            return key
        key = f"status:{level}"
        labels[key] = {"kind": "status", "name": level}
        return key

    actual_labels = [species_label(taxon_id) for taxon_id in sorted(truth_names)]
    overall: dict[str, Counter] = defaultdict(Counter)
    per_type: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in details:
        actual = _species_key(row["truth_species_id"])
        predicted = prediction_label(row)
        overall[actual][predicted] += 1
        per_type[row["clue_type"]][actual][predicted] += 1

    # All true species are also prediction columns, even when never predicted.
    # Extra species, higher taxa, and statuses follow; no predictions are discarded.
    predicted_labels = actual_labels + sorted(set(labels) - set(actual_labels))

    def matrix(counts: dict[str, Counter]) -> dict:
        ordered = {
            actual: {key: counts[actual][key] for key in sorted(counts[actual])}
            for actual in actual_labels
            if actual in counts
        }
        row_totals = {actual: sum(values.values()) for actual, values in ordered.items()}
        return {
            "total": sum(row_totals.values()),
            "exact": sum(values.get(actual, 0) for actual, values in ordered.items()),
            "row_totals": row_totals,
            "counts": ordered,
        }

    return {
        "format_version": 1,
        "orientation": "rows=actual, columns=predicted",
        "normalization": "raw_counts",
        "labels": labels,
        "actual_labels": actual_labels,
        "predicted_labels": predicted_labels,
        "overall": matrix(overall),
        "per_clue_type": {kind: matrix(per_type[kind]) for kind in sorted(per_type)},
    }


def write_confusion_matrix(score_dir: str | Path) -> Path:
    """Derive a report without model calls or changes to existing score records."""
    directory = Path(score_dir)
    scores_bytes = (directory / "scores.jsonl").read_bytes()
    summary_bytes = (directory / "summary.json").read_bytes()
    details = [json.loads(line) for line in scores_bytes.splitlines() if line.strip()]
    summary = json.loads(summary_bytes)
    if dict(Counter(row["level"] for row in details)) != summary["overall"]:
        raise ValueError("scores and summary outcome counts differ")
    per_type: dict[str, Counter] = defaultdict(Counter)
    for row in details:
        per_type[row["clue_type"]][row["level"]] += 1
    if dict(per_type) != summary["per_type"]:
        raise ValueError("scores and summary per-trace counts differ")
    report = {
        **build_confusion_matrix(details),
        "model_id": summary["model_id"],
        "suite": summary["suite"],
        "tasks_hash": summary["tasks_hash"],
        "predictions_sha256": summary["predictions_sha256"],
        "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "score_complete": summary["score_complete"],
        "provisional_due_to_extractor": summary["provisional_due_to_extractor"],
        "review_required": summary["review_required"],
    }
    path = directory / "confusion_matrix.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path
