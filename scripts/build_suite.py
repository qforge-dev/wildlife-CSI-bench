"""Build a frozen Wildlife CSI suite using AnimalClue observation taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from wildlife_csi.builder import HuggingFaceAnimalClue, SuiteBuilder  # noqa: E402
from wildlife_csi.animalclue import AccessPending, REPOS, require_access  # noqa: E402
from wildlife_csi.suite import PER_TYPE, SEED  # noqa: E402
from wildlife_csi.source import INaturalist  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=PER_TYPE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--clues", nargs="+", default=list(REPOS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, help="parallel image/label downloads")
    args = parser.parse_args()
    try:
        require_access(args.clues)
    except AccessPending as exc:
        parser.exit(1, f"{exc}\n")
    result = SuiteBuilder(HuggingFaceAnimalClue(), INaturalist(), max_workers=args.workers).build(
        args.limit, args.seed, args.clues, args.out
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
