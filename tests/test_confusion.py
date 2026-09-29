import hashlib
import json
from collections import Counter

import pytest
from typer.testing import CliRunner

from wildlife_csi.cli import app
from wildlife_csi.confusion import build_confusion_matrix, write_confusion_matrix


def row(task_id, truth_id=1, level="exact", resolution=None, clue="egg"):
    return {
        "task_id": task_id,
        "truth_species_id": truth_id,
        "truth": {1: "Vulpes vulpes", 2: "Canis lupus"}[truth_id],
        "level": level,
        "resolution": resolution,
        "clue_type": clue,
    }


def test_counts_preserve_all_outcomes_and_taxonomy_identity():
    details = [
        row("canonical"),
        row(
            "common-name", resolution={"species_id": 1, "taxon": "Vulpes vulpes", "rank": "species"}
        ),
        row(
            "subspecies",
            resolution={"species_id": 1, "taxon": "Vulpes vulpes fulva", "rank": "subspecies"},
        ),
        row(
            "fox-as-wolf",
            level="wrong",
            resolution={"species_id": 2, "taxon": "Canis lupus", "rank": "species"},
        ),
        row(
            "wolf-as-fox",
            truth_id=2,
            level="wrong",
            resolution={"species_id": 1, "taxon": "Vulpes vulpes", "rank": "species"},
        ),
        row(
            "outside-truth",
            level="wrong",
            resolution={"species_id": 99, "taxon": "Panthera tigris altaica", "rank": "subspecies"},
        ),
        row(
            "genus",
            level="genus_only",
            resolution={"taxon_id": 10, "taxon": "Vulpes", "rank": "genus"},
        ),
    ]
    for status in [
        "abstain",
        "unresolved",
        "invalid",
        "ambiguous",
        "empty",
        "truncated",
        "transport_error",
        "refused",
        "missing",
    ]:
        details.append(row(status, truth_id=2, level=status, clue="footprint"))
    report = build_confusion_matrix(details)
    matrix = report["overall"]
    assert report["actual_labels"] == ["species:1", "species:2"]
    assert report["predicted_labels"][:2] == report["actual_labels"]
    assert matrix["total"] == 16
    assert matrix["exact"] == 3
    assert matrix["row_totals"] == {"species:1": 6, "species:2": 10}
    assert matrix["counts"]["species:1"] == {
        "species:1": 3,
        "species:2": 1,
        "species:99": 1,
        "taxon:10": 1,
    }
    assert "species:2" not in matrix["counts"]["species:2"]  # zero-correct species retained
    assert matrix["counts"]["species:2"]["status:transport_error"] == 1
    assert matrix["counts"]["species:2"]["status:abstain"] == 1
    assert report["labels"]["species:1"]["name"] == "Vulpes vulpes"
    assert report["labels"]["species:99"]["name"] is None
    assert report["labels"]["species:99"]["observed_names"] == ["Panthera tigris altaica"]
    assert report["labels"]["taxon:10"]["kind"] == "higher_taxon"
    assert report["per_clue_type"]["egg"]["total"] == 7
    assert report["per_clue_type"]["footprint"]["total"] == 9
    assert build_confusion_matrix(list(reversed(details))) == report


@pytest.mark.parametrize(
    "details, message",
    [
        ([], "without scores"),
        ([row("same"), row("same")], "duplicate task"),
        ([row("bad", resolution={"species_id": 2})], "exact score conflicts"),
        ([row("bad", level="wrong", resolution={"species_id": 1})], "non-exact score"),
    ],
)
def test_rejects_records_that_would_misstate_the_matrix(details, message):
    with pytest.raises(ValueError, match=message):
        build_confusion_matrix(details)


def write_inputs(directory):
    directory.mkdir()
    details = [row("right"), row("blocked", level="transport_error")]
    summary = {
        "model_id": "example",
        "suite": "test-suite",
        "tasks_hash": "tasks-hash",
        "predictions_sha256": "prediction-hash",
        "score_complete": True,
        "provisional_due_to_extractor": True,
        "review_required": 1,
        "overall": dict(Counter(r["level"] for r in details)),
        "per_type": {"egg": {"exact": 1, "transport_error": 1}},
    }
    (directory / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
    (directory / "summary.json").write_text(json.dumps(summary))
    return summary


def test_cli_backfills_multiple_runs_with_provenance_and_without_changing_inputs(tmp_path):
    directories = [tmp_path / "one", tmp_path / "two"]
    for directory in directories:
        write_inputs(directory)
    before = {path: path.read_bytes() for directory in directories for path in directory.iterdir()}
    result = CliRunner().invoke(app, ["confusion", *map(str, directories)])
    assert result.exit_code == 0, result.output
    for directory in directories:
        path = directory / "confusion_matrix.json"
        first = path.read_bytes()
        report = json.loads(first)
        assert report["overall"]["total"] == 2
        assert report["overall"]["exact"] == 1
        assert (
            report["scores_sha256"]
            == hashlib.sha256(before[directory / "scores.jsonl"]).hexdigest()
        )
        assert (
            report["summary_sha256"]
            == hashlib.sha256(before[directory / "summary.json"]).hexdigest()
        )
        assert report["provisional_due_to_extractor"] is True
        assert report["review_required"] == 1
        write_confusion_matrix(directory)
        assert path.read_bytes() == first
    assert all(path.read_bytes() == content for path, content in before.items())


def test_backfill_rejects_scores_from_a_different_summary(tmp_path):
    directory = tmp_path / "score"
    summary = write_inputs(directory)
    summary["overall"] = {"exact": 2}
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="outcome counts differ"):
        write_confusion_matrix(directory)
    assert not (directory / "confusion_matrix.json").exists()
