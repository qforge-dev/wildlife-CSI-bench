"""Visual QA replacements for a frozen AnimalClue suite.

The curated suite is written to a new directory. Every replacement must be an
accepted, previously unselected candidate from the original selection ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from wildlife_csi.animalclue import REPOS
from wildlife_csi.builder import IMG_EXTS, SuiteBuilder
from wildlife_csi.selection import selection_stats
from wildlife_csi.source import photo_reference
from wildlife_csi.suite import validate_suite, write_manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SuiteCurator:
    def __init__(self, builder: SuiteBuilder):
        self.builder = builder

    def _candidate(self, clue: str, obs: str, files: list[str]) -> tuple[dict, bytes]:
        labels = {
            Path(path).stem: path for path in files if "/test/" in path and path.endswith(".txt")
        }
        image_labels = [
            (path, labels[Path(path).stem])
            for path in files
            if "/test/" in path
            and path.lower().endswith(IMG_EXTS)
            and photo_reference(path)[0] == obs
            and Path(path).stem in labels
        ]
        if not image_labels:
            raise ValueError(f"candidate observation {obs} has no labeled photos")
        observation = self.builder.taxonomy.observations([obs])[obs]
        taxon_id = int(observation["taxon"]["id"])
        taxon = self.builder.taxonomy.taxa([taxon_id])[taxon_id]
        return self.builder._prepare_observation(
            clue, REPOS[clue], obs, image_labels, observation, taxon
        )

    def curate(self, source: Path, out: Path, replacements: dict[str, str]) -> dict:
        if out.exists():
            raise FileExistsError(f"curated suite already exists: {out}")
        tasks, source_manifest = validate_suite(source / "tasks.jsonl", source / "images")
        ledger = _read_jsonl(source / "selection.jsonl")
        exclusions = _read_jsonl(source / "exclusions.jsonl")
        selected_by_obs = {t["source"]["observation_id"]: t for t in tasks}
        candidate_by_key = {(row["clue_type"], row["observation_id"]): row for row in ledger}
        if len(candidate_by_key) != len(ledger):
            raise ValueError("selection ledger has duplicate clue/observation pairs")
        if len(set(replacements.values())) != len(replacements):
            raise ValueError("replacement observations must be distinct")
        for old_obs, new_obs in replacements.items():
            if old_obs not in selected_by_obs:
                raise ValueError(f"observation {old_obs} is not selected")
            if new_obs in selected_by_obs:
                raise ValueError(f"observation {new_obs} is already selected")
            clue = selected_by_obs[old_obs]["clue_type"]
            candidate = candidate_by_key.get((clue, new_obs))
            if candidate is None or candidate["selected"]:
                raise ValueError(f"observation {new_obs} is not an unselected candidate")

        by_clue: dict[str, list[str]] = {}
        for old_obs in replacements:
            clue = selected_by_obs[old_obs]["clue_type"]
            by_clue.setdefault(clue, self.builder.repository.list_files(REPOS[clue]))

        new_tasks = list(tasks)
        payloads: dict[str, bytes] = {}
        swaps = []
        for old_obs, new_obs in replacements.items():
            old = selected_by_obs[old_obs]
            clue = old["clue_type"]
            task, payload = self._candidate(clue, new_obs, by_clue[clue])
            ledger_row = candidate_by_key[(clue, new_obs)]
            if (
                task["task_id"] != ledger_row["task_id"]
                or task["image_sha256"] != ledger_row["image_sha256"]
                or task["correct_taxon_id"] != ledger_row["species_id"]
                or task["source"]["quality"] != ledger_row["quality"]
            ):
                raise ValueError(f"candidate {new_obs} differs from frozen selection ledger")
            new_tasks[new_tasks.index(old)] = task
            payloads[task["image_sha256"]] = payload
            swaps.append(
                {
                    "clue": clue,
                    "rejected_observation": old_obs,
                    "replacement_observation": new_obs,
                    "reason": "visual_qa_no_visible_trace",
                }
            )
            candidate_by_key[(clue, old_obs)]["selected"] = False
            candidate_by_key[(clue, old_obs)]["visual_qa"] = "rejected_no_visible_trace"
            ledger_row["selected"] = True
            ledger_row["visual_qa"] = "approved_replacement"

        if len({t["image_sha256"] for t in new_tasks}) != len(new_tasks):
            raise ValueError("curated suite has duplicate images")
        replacement_obs = set(replacements.values())
        exclusions = [
            row
            for row in exclusions
            if not (
                row["obs"] in replacement_obs
                and row["reason"] == "not_selected_after_quality_diversity"
            )
        ]
        exclusions.extend(
            {"clue": row["clue"], "obs": row["rejected_observation"], "reason": row["reason"]}
            for row in swaps
        )
        selection_bytes = (
            "\n".join(json.dumps(row, sort_keys=True) for row in ledger) + "\n"
        ).encode()
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as staging:
            stage = Path(staging)
            (stage / "images").mkdir()
            for task in new_tasks:
                sha = task["image_sha256"]
                image = stage / "images" / f"{sha}.jpg"
                if sha in payloads:
                    image.write_bytes(payloads[sha])
                else:
                    os.link(source / "images" / f"{sha}.jpg", image)
            (stage / "selection.jsonl").write_bytes(selection_bytes)
            manifest = write_manifest(stage, new_tasks, source_manifest["seed"], exclusions)
            manifest["selection"] = {
                "policy": source_manifest["selection"]["policy"] + "+visual-qa-v1",
                "candidate_multiplier": source_manifest["selection"]["candidate_multiplier"],
                "ledger_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                "per_type": {
                    clue: {
                        **selection_stats([], [t for t in new_tasks if t["clue_type"] == clue]),
                        "candidates": sum(row["clue_type"] == clue for row in ledger),
                    }
                    for clue in source_manifest["per_type"]
                },
            }
            manifest["curation"] = {
                "source_tasks_hash": source_manifest["tasks_hash"],
                "replacements": swaps,
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            validate_suite(stage / "tasks.jsonl", stage / "images")
            stage.rename(out)
        return manifest
