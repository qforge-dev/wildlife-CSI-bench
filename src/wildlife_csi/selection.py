"""Deterministic quality ranking and species-diverse suite selection."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

from PIL import Image, ImageChops, ImageFilter, ImageStat


CANDIDATE_MULTIPLIER = 2.0
SELECTION_POLICY = "quality-with-species-rounds-v3"


def image_quality(image: Image.Image) -> dict:
    """Rank technically usable crops; preserve the measurements for audit.

    Resolution is the main signal. Contrast and local detail only break ties;
    neither is used as a semantic claim that the trace is identifiable.
    """
    short_side = min(image.size)
    sample = image.convert("L")
    sample.thumbnail((256, 256), Image.Resampling.LANCZOS)
    contrast = ImageStat.Stat(sample).stddev[0]
    detail = ImageStat.Stat(
        ImageChops.difference(sample, sample.filter(ImageFilter.GaussianBlur(radius=2)))
    ).mean[0]
    score = (
        0.7 * min(math.log2(max(short_side, 320) / 320), 2.0) / 2.0
        + 0.2 * min(contrast / 64, 1.0)
        + 0.1 * min(detail / 18, 1.0)
    )
    return {
        "native_crop_short_side": short_side,
        "contrast": round(contrast, 3),
        "detail": round(detail, 3),
        "score": round(score, 6),
    }


def quality_rejection(clue: str, quality: dict) -> str | None:
    """Exclude crops unlikely to support a visual identification."""
    if quality["native_crop_short_side"] < 400:
        return "crop_resolution_below_400"
    if quality["contrast"] < 20:
        return "low_contrast_crop"
    if quality["detail"] < (4 if clue == "footprint" else 2.5):
        return "low_detail_crop"
    if quality["score"] < 0.4:
        return "quality_score_below_0_4"
    return None


def select_diverse(
    candidates: list[dict],
    limit: int,
    seed: int,
    clue: str,
    used_observations: set[str] | None = None,
    used_images: set[str] | None = None,
) -> list[dict]:
    """Take each species' best crop, then second best, and so on."""
    used_obs = set(used_observations or ())
    used_sha = set(used_images or ())
    groups: dict[int, list[dict]] = defaultdict(list)
    for task in candidates:
        groups[task["correct_taxon_id"]].append(task)
    for group in groups.values():
        group.sort(
            key=lambda t: (
                -t["source"]["quality"]["score"],
                hashlib.sha256(
                    f"{seed}:{clue}:{t['source']['observation_id']}".encode()
                ).hexdigest(),
            )
        )
    species = sorted(
        groups, key=lambda sid: hashlib.sha256(f"{seed}:{clue}:{sid}".encode()).hexdigest()
    )
    selected: list[dict] = []
    depth = 0
    while len(selected) < limit and any(depth < len(groups[sid]) for sid in species):
        for sid in species:
            if depth >= len(groups[sid]):
                continue
            task = groups[sid][depth]
            obs = task["source"]["observation_id"]
            sha = task["image_sha256"]
            if obs not in used_obs and sha not in used_sha:
                selected.append(task)
                used_obs.add(obs)
                used_sha.add(sha)
                if len(selected) == limit:
                    break
        depth += 1
    if len(selected) != limit:
        raise ValueError(f"{clue}: only {len(selected)}/{limit} distinct usable candidates")
    return selected


def selection_stats(candidates: list[dict], selected: list[dict]) -> dict:
    counts = Counter(t["correct_taxon_id"] for t in selected)
    return {
        "candidates": len(candidates),
        "selected": len(selected),
        "species": len(counts),
        "max_per_species": max(counts.values(), default=0),
        "top_ten_species_images": sum(n for _, n in counts.most_common(10)),
        "quality_score_min": min((t["source"]["quality"]["score"] for t in selected), default=None),
        "quality_score_median": (
            sorted(t["source"]["quality"]["score"] for t in selected)[len(selected) // 2]
            if selected
            else None
        ),
    }
