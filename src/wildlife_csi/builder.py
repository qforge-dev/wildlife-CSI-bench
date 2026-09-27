"""Suite selection, trace cropping, and source-label assembly.

Repository access, taxonomy lookup, and image preparation are injected so
source providers and normalization policies can be upgraded independently.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from wildlife_csi.animalclue import REPOS, crop_box, parse_seg_label, parse_yolo_label, seg_bbox
from wildlife_csi.selection import (
    CANDIDATE_MULTIPLIER,
    SELECTION_POLICY,
    image_quality,
    quality_rejection,
    select_diverse,
    selection_stats,
)
from wildlife_csi.suite import PROTOCOL_VERSION, build_task, prepare_image, write_manifest
from wildlife_csi.location_enrichment import observation_country
from wildlife_csi.source import photo_reference, source_record

IMG_EXTS = (".jpg", ".jpeg", ".png")


def trace_box(clue: str, label: str, width: int, height: int) -> tuple[int, tuple, int]:
    """Crop the largest valid annotated trace in a source image."""
    if clue == "footprint":
        annotations = parse_yolo_label(label)
        if not annotations or len({row[0] for row in annotations}) != 1:
            raise ValueError("empty_or_conflicting_annotation")
        valid = [
            row
            for row in annotations
            if all(math.isfinite(v) for v in row[1:])
            and 0 <= row[1] <= 1
            and 0 <= row[2] <= 1
            and 0 < row[3] <= 1
            and 0 < row[4] <= 1
        ]
        if not valid:
            raise ValueError("invalid_bbox")
        chosen = max(valid, key=lambda row: row[3] * row[4])
        box = crop_box(width, height, *chosen[1:])
        class_id = chosen[0]
    else:
        annotations = parse_seg_label(label)
        if not annotations or len({row[0] for row in annotations}) != 1:
            raise ValueError("empty_or_conflicting_annotation")
        valid = [
            row
            for row in annotations
            if len(row[1]) >= 6
            and len(row[1]) % 2 == 0
            and all(math.isfinite(v) and 0 <= v <= 1 for v in row[1])
        ]
        if not valid:
            raise ValueError("invalid_polygon")
        chosen = max(
            valid,
            key=lambda row: (max(row[1][0::2]) - min(row[1][0::2]))
            * (max(row[1][1::2]) - min(row[1][1::2])),
        )
        box = seg_bbox(chosen[1], width, height)
        class_id = chosen[0]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("invalid_crop_box")
    return class_id, box, len(annotations)


class HuggingFaceAnimalClue:
    """Adapter for pinned, gated AnimalClue dataset repositories."""

    def __init__(self, api=None, downloader=None, token: str | None = None):
        from huggingface_hub import HfApi, hf_hub_download

        token_path = Path.home() / ".cache/huggingface/token"
        self.token = (
            token
            if token is not None
            else (token_path.read_text().strip() if token_path.exists() else None)
        )
        self.api = api or HfApi(token=self.token)
        self.downloader = downloader or hf_hub_download
        self._rate_lock = threading.Lock()
        self._blocked_until = 0.0

    def _with_rate_limit_retry(self, call: Callable):
        from huggingface_hub.errors import HfHubHTTPError

        for attempt in range(8):
            with self._rate_lock:
                wait = max(0.0, self._blocked_until - time.monotonic())
            if wait:
                time.sleep(wait)
            try:
                return call()
            except HfHubHTTPError as exc:
                if exc.response is None or exc.response.status_code != 429 or attempt == 7:
                    raise
                header = exc.response.headers.get("Retry-After", "")
                try:
                    delay = max(60.0, float(header))
                except ValueError:
                    delay = 90.0
                with self._rate_lock:
                    self._blocked_until = max(self._blocked_until, time.monotonic() + delay)
        raise AssertionError("unreachable")

    def list_files(self, spec: dict) -> list[str]:
        return self._with_rate_limit_retry(
            lambda: self.api.list_repo_files(
                spec["repo"], repo_type="dataset", revision=spec["revision"]
            )
        )

    def download(self, spec: dict, path: str) -> Path:
        return Path(
            self._with_rate_limit_retry(
                lambda: self.downloader(
                    spec["repo"],
                    path,
                    repo_type="dataset",
                    revision=spec["revision"],
                    token=self.token,
                )
            )
        )


class SuiteBuilder:
    def __init__(
        self,
        repository,
        taxonomy,
        image_preparer: Callable = prepare_image,
        quality_scorer: Callable = image_quality,
        max_workers: int = 1,
        location_resolver: Callable | None = None,
    ):
        self.repository = repository
        self.taxonomy = taxonomy
        self.image_preparer = image_preparer
        self.quality_scorer = quality_scorer
        self.location_resolver = location_resolver or self._resolve_public_location
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def _resolve_public_location(self, observation: dict, source: dict) -> dict:
        place_ids = observation.get("place_ids") or []
        places = self.taxonomy.places(place_ids)
        indexed = {
            int(pid): {
                "found": int(pid) in places,
                "name": places[int(pid)].get("name") if int(pid) in places else None,
                "admin_level": places[int(pid)].get("admin_level") if int(pid) in places else None,
                "place_type": places[int(pid)].get("place_type") if int(pid) in places else None,
            }
            for pid in place_ids
        }
        location = observation_country(observation, indexed)
        if location["country"] is None:
            raise ValueError("source observation has no public country")
        location["observation_id"] = source["observation_id"]
        return location

    def _prepare_candidate(
        self,
        clue: str,
        spec: dict,
        obs: str,
        image_path: str,
        label_path: str,
        observation: dict,
        taxon: dict,
    ) -> tuple[dict, bytes]:
        truth = source_record(observation, taxon, image_path)
        image_file = self.repository.download(spec, image_path)
        label_file = self.repository.download(spec, label_path)
        with Image.open(image_file) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            class_id, box, annotation_count = trace_box(
                clue, label_file.read_text(), img.width, img.height
            )
            cropped = img.crop(box)
            quality = self.quality_scorer(cropped)
            if reason := quality_rejection(clue, quality):
                raise ValueError(reason)
            buf = io.BytesIO()
            cropped.save(buf, "PNG")
        payload, meta = self.image_preparer(buf.getvalue())
        source = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "split": "test",
            "observation_id": obs,
            "label_path": label_path,
            "image_path": image_path,
            "yolo_class_id": class_id,
            "annotation_count": annotation_count,
            "crop_box": list(box),
            "quality": quality,
            "prepared": {"width": meta["width"], "height": meta["height"]},
            "inat": truth,
        }
        return build_task(
            clue, meta["sha256"], source, self.location_resolver(observation, source)
        ), payload

    def _prepare_observation(
        self,
        clue: str,
        spec: dict,
        obs: str,
        image_labels: list[tuple[str, str]],
        observation: dict,
        taxon: dict,
    ) -> tuple[dict, bytes]:
        """Keep the strongest eligible source photo for one observation."""
        best: tuple[dict, bytes] | None = None
        failures: list[str] = []
        accepted = 0
        for image_path, label_path in image_labels:
            try:
                task, payload = self._prepare_candidate(
                    clue, spec, obs, image_path, label_path, observation, taxon
                )
            except (ValueError, KeyError, IndexError) as exc:
                failures.append(str(exc))
                continue
            accepted += 1
            if (
                best is None
                or task["source"]["quality"]["score"] > best[0]["source"]["quality"]["score"]
            ):
                best = task, payload
        if best is None:
            reason = Counter(failures).most_common(1)[0][0] if failures else "no_labeled_photo"
            raise ValueError(f"no_usable_photo:{reason}")
        best[0]["source"]["observation_photo_candidates"] = len(image_labels)
        best[0]["source"]["observation_photos_accepted"] = accepted
        return best

    def build(self, limit: int, seed: int, clues: list[str], out: Path) -> dict:
        if limit < 1 or not clues or len(set(clues)) != len(clues):
            raise ValueError("limit must be positive and clues unique/nonempty")
        if unknown := set(clues) - REPOS.keys():
            raise ValueError(f"unknown clues: {sorted(unknown)}")
        if (out / "manifest.json").exists():
            raise FileExistsError(f"frozen suite already exists: {out}; choose a new --out")
        (out / "images").mkdir(parents=True, exist_ok=True)
        checkpoint_meta = out / "build-checkpoint.json"
        checkpoint_rows = out / "build-checkpoint.jsonl"
        expected_meta = {
            "limit": limit,
            "seed": seed,
            "clues": clues,
            "protocol_version": PROTOCOL_VERSION,
            "revisions": {clue: REPOS[clue]["revision"] for clue in clues},
            "selection_policy": SELECTION_POLICY,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
        }
        if checkpoint_meta.exists():
            if json.loads(checkpoint_meta.read_text()) != expected_meta:
                raise ValueError("build checkpoint settings differ; choose a new --out")
        elif checkpoint_rows.exists():
            raise ValueError("build checkpoint metadata is missing; choose a new --out")
        else:
            checkpoint_meta.write_text(json.dumps(expected_meta, sort_keys=True) + "\n")
        candidates, exclusions = [], []
        completed: dict[str, set[str]] = {clue: set() for clue in clues}
        if checkpoint_rows.exists():
            for line in checkpoint_rows.read_text().splitlines():
                event = json.loads(line)
                row = event["row"]
                if event["kind"] == "candidate":
                    clue, obs = row["clue_type"], row["source"]["observation_id"]
                else:
                    clue, obs = row["clue"], row["obs"]
                if clue not in completed or obs in completed[clue]:
                    raise ValueError("duplicate or invalid build checkpoint observation")
                if event["kind"] == "candidate":
                    image = out / "images" / f"{row['image_sha256']}.jpg"
                    if (
                        not image.exists()
                        or hashlib.sha256(image.read_bytes()).hexdigest() != row["image_sha256"]
                    ):
                        raise ValueError(f"build checkpoint image missing or corrupt: {image}")
                    candidates.append(row)
                elif event["kind"] == "exclusion":
                    exclusions.append(row)
                else:
                    raise ValueError("invalid build checkpoint event")
                completed[clue].add(obs)

        def record(kind: str, row: dict) -> None:
            with checkpoint_rows.open("a") as stream:
                stream.write(json.dumps({"kind": kind, "row": row}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        def exclude(clue: str, obs: str, reason: str) -> None:
            row = {"clue": clue, "obs": obs, "reason": reason}
            record("exclusion", row)
            exclusions.append(row)
            completed[clue].add(obs)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for clue in clues:
                picked = sum(task["clue_type"] == clue for task in candidates)
                spec = REPOS[clue]
                files = self.repository.list_files(spec)
                images = sorted(f for f in files if "/test/" in f and f.lower().endswith(IMG_EXTS))
                labels = {Path(f).stem: f for f in files if "/test/" in f and f.endswith(".txt")}
                by_obs: dict[str, list[str]] = {}
                for path in images:
                    try:
                        obs_id, _ = photo_reference(path)
                    except ValueError:
                        continue
                    by_obs.setdefault(obs_id, []).append(path)
                obs_ids = sorted(by_obs)
                random.Random(f"{seed}:{clue}").shuffle(obs_ids)
                target = min(len(obs_ids), max(limit, math.ceil(limit * CANDIDATE_MULTIPLIER)))
                for offset in range(0, len(obs_ids), 20):
                    if picked >= target:
                        break
                    batch = [
                        obs for obs in obs_ids[offset : offset + 20] if obs not in completed[clue]
                    ]
                    if not batch:
                        continue
                    obs_rows = self.taxonomy.observations(batch, require_all=False)
                    taxon_ids = [
                        int(obs_rows[obs]["taxon"]["id"])
                        for obs in batch
                        if obs in obs_rows and obs_rows[obs].get("taxon")
                    ]
                    taxa = self.taxonomy.taxa(taxon_ids)
                    pending = []
                    for obs in batch:
                        if obs not in obs_rows or not obs_rows[obs].get("taxon"):
                            exclude(clue, obs, "source_observation_missing_taxon")
                            continue
                        image_labels = [
                            (path, label)
                            for path in by_obs[obs]
                            if (label := labels.get(Path(path).stem)) is not None
                        ]
                        if not image_labels:
                            exclude(clue, obs, "label_not_found")
                            continue
                        taxon = taxa[int(obs_rows[obs]["taxon"]["id"])]
                        future = executor.submit(
                            self._prepare_observation,
                            clue,
                            spec,
                            obs,
                            image_labels,
                            obs_rows[obs],
                            taxon,
                        )
                        pending.append((obs, future))
                    for obs, future in pending:
                        try:
                            task, payload = future.result()
                        except (ValueError, KeyError, IndexError) as exc:
                            exclude(clue, obs, str(exc))
                            continue
                        if picked >= target:
                            continue
                        (out / "images" / f"{task['image_sha256']}.jpg").write_bytes(payload)
                        record("candidate", task)
                        candidates.append(task)
                        completed[clue].add(obs)
                        picked += 1
                        print(
                            f"[{clue}] candidate {picked}/{target} obs={obs} species={task['correct_taxon']}",
                            flush=True,
                        )
                if picked < limit:
                    raise RuntimeError(
                        f"{clue}: only {picked}/{limit} usable candidates; suite not frozen"
                    )

        tasks: list[dict] = []
        selected_by_clue: dict[str, list[dict]] = {}
        for clue in clues:
            pool = [t for t in candidates if t["clue_type"] == clue]
            chosen = select_diverse(
                pool,
                limit,
                seed,
                clue,
                {t["source"]["observation_id"] for t in tasks},
                {t["image_sha256"] for t in tasks},
            )
            selected_by_clue[clue] = chosen
            tasks.extend(chosen)
        chosen_keys = {(t["clue_type"], t["source"]["observation_id"]) for t in tasks}
        ledger = [
            {
                "task_id": t["task_id"],
                "clue_type": t["clue_type"],
                "observation_id": t["source"]["observation_id"],
                "species_id": t["correct_taxon_id"],
                "species": t["correct_taxon"],
                "image_sha256": t["image_sha256"],
                "quality": t["source"]["quality"],
                "selected": (t["clue_type"], t["source"]["observation_id"]) in chosen_keys,
            }
            for t in candidates
        ]
        selection_bytes = (
            "\n".join(json.dumps(row, sort_keys=True) for row in ledger) + "\n"
        ).encode()
        (out / "selection.jsonl").write_bytes(selection_bytes)
        for t in candidates:
            if (t["clue_type"], t["source"]["observation_id"]) not in chosen_keys:
                exclusions.append(
                    {
                        "clue": t["clue_type"],
                        "obs": t["source"]["observation_id"],
                        "reason": "not_selected_after_quality_diversity",
                    }
                )
        manifest = write_manifest(out, tasks, seed, exclusions)
        manifest["selection"] = {
            "policy": SELECTION_POLICY,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "ledger_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "per_type": {
                clue: selection_stats(
                    [t for t in candidates if t["clue_type"] == clue], selected_by_clue[clue]
                )
                for clue in clues
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        selected_images = {t["image_sha256"] for t in tasks}
        for t in candidates:
            if t["image_sha256"] not in selected_images:
                (out / "images" / f"{t['image_sha256']}.jpg").unlink(missing_ok=True)
        checkpoint_rows.unlink(missing_ok=True)
        checkpoint_meta.unlink(missing_ok=True)
        return manifest
