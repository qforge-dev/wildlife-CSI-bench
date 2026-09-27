"""Evaluate the answer extractor on explicit text-only cases, retaining all calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dotenv

from wildlife_csi.extractor_eval import evaluate_file
from wildlife_csi.answer_extractor import load_extractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="tests/fixtures/answer_extractor_cases.jsonl")
    parser.add_argument("--config", default="configs/extractors/answer.yaml")
    parser.add_argument("--out", default="data/work/extractor-validation")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    dotenv.load_dotenv(Path(".env"))
    print(
        json.dumps(
            evaluate_file(args.cases, args.out, load_extractor(args.config), offline=args.offline),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
