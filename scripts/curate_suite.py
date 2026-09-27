"""Apply documented visual QA replacements to the frozen 2,000-image suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from wildlife_csi.builder import HuggingFaceAnimalClue, SuiteBuilder  # noqa: E402
from wildlife_csi.curation import SuiteCurator  # noqa: E402
from wildlife_csi.source import INaturalist  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD_OBS:NEW_OBS",
        help="replace an invalid selected image with a reviewed candidate",
    )
    args = parser.parse_args()
    if not args.replace:
        parser.error("at least one --replace is required")
    try:
        pairs = [item.split(":", 1) for item in args.replace]
        if any(len(pair) != 2 or not all(pair) for pair in pairs):
            raise ValueError
    except ValueError:
        parser.error("--replace must be OLD_OBS:NEW_OBS")
    if len({old for old, _ in pairs}) != len(pairs):
        parser.error("each rejected observation can appear only once")
    builder = SuiteBuilder(HuggingFaceAnimalClue(), INaturalist())
    manifest = SuiteCurator(builder).curate(args.source, args.out, dict(pairs))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
