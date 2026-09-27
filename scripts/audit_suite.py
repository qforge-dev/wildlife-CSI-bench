"""Summarize a frozen suite's species spread, image quality, and source rights."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from wildlife_csi.suite import validate_suite  # noqa: E402
from wildlife_csi.s3_suite import S3ImageStore  # noqa: E402
from wildlife_csi.storage import LocalImageStore  # noqa: E402


def write_contact_sheet(tasks: list[dict], image_store, dest: Path, seed: int) -> None:
    sample = random.Random(seed).sample(tasks, min(40, len(tasks)))
    sheet = Image.new("RGB", (1200, 180 * ((len(sample) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, task in enumerate(sample):
        x, y = (index % 4) * 300, (index // 4) * 180
        with Image.open(io.BytesIO(image_store.load(task))) as source:
            image = source.copy()
        image.thumbnail((286, 148))
        sheet.paste(image, (x + (286 - image.width) // 2, y))
        draw.text((x + 4, y + 151), task["correct_taxon"][:38], fill="black")
        draw.text((x + 4, y + 165), task["source"]["observation_id"], fill="black")
    sheet.save(dest, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/work/audit"))
    args = parser.parse_args()
    root = args.suite
    tasks, manifest = validate_suite(root / "tasks.jsonl")
    image_store = None
    if args.contact_sheets:
        image_store = (
            S3ImageStore(region=manifest["storage"]["region"])
            if manifest.get("storage", {}).get("kind") == "s3"
            else LocalImageStore(root / "images")
        )
        args.out.mkdir(parents=True, exist_ok=True)
    exclusions = [
        json.loads(line)
        for line in (root / "exclusions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report = {
        "tasks_hash": manifest["tasks_hash"],
        "total": len(tasks),
        "per_type": {},
        "photo_licenses": dict(
            Counter(task["source"]["inat"].get("photo_license") or "missing" for task in tasks)
        ),
        "missing_photo_attribution": sum(
            not task["source"]["inat"].get("photo_attribution") for task in tasks
        ),
    }
    for clue in manifest["per_type"]:
        rows = [task for task in tasks if task["clue_type"] == clue]
        if not rows:
            continue
        species = Counter(task["correct_taxon_id"] for task in rows)
        scores = sorted(task["source"]["quality"]["score"] for task in rows)
        report["per_type"][clue] = {
            "images": len(rows),
            "species": len(species),
            "max_per_species": max(species.values()),
            "top_ten_species_images": sum(count for _, count in species.most_common(10)),
            "selected_alternative_photo": sum(
                task["source"]["inat"]["photo_index"] > 0 for task in rows
            ),
            "observations_with_multiple_labeled_photos": sum(
                task["source"].get("observation_photo_candidates", 1) > 1 for task in rows
            ),
            "quality_min": scores[0],
            "quality_median": scores[len(scores) // 2],
            "exclusions": dict(Counter(row["reason"] for row in exclusions if row["clue"] == clue)),
        }
        if args.contact_sheets:
            write_contact_sheet(rows, image_store, args.out / f"qa-{clue}.jpg", manifest["seed"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
