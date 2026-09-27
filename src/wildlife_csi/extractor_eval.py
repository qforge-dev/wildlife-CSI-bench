"""Response-only checks for the official answer extractor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wildlife_csi.answer_extractor import answer_sha256
from wildlife_csi.ports import AnswerExtractor, ExtractionStore
from wildlife_csi.provider_record import ProviderCallError


def evaluate_cases(
    cases: list[dict], extractor: AnswerExtractor, store: ExtractionStore, *, offline: bool = False
) -> tuple[dict, list[dict]]:
    rows = []
    for case in cases:
        raw = case["raw"]
        cached = store.get(case["id"])
        if cached is not None and cached["raw_answer_sha256"] != answer_sha256(raw):
            raise ValueError(f"cached extractor input changed: {case['id']}")
        if cached is None:
            if offline:
                raise ValueError(f"extractor result not cached: {case['id']}")
            try:
                extraction = extractor.extract(raw)
            except ProviderCallError as exc:
                extraction = {
                    "status": "provider_error",
                    "error": str(exc),
                    "provider": exc.details,
                }
            cached = {
                "task_id": case["id"],
                "raw_answer_sha256": answer_sha256(raw),
                "extraction": extraction,
            }
            store.put(cached)
        extraction = cached["extraction"]
        decision = extraction.get("decision") or {}
        expected = case["decision"]
        actual = decision.get("decision") if extraction.get("status") == "ok" else None
        span = decision.get("source_span", "")
        exact_decision = actual == expected and (
            expected != "single" or span.casefold() in [x.casefold() for x in case["spans"]]
        )
        scoring_safe = exact_decision if expected == "single" else actual in {"ambiguous", "none"}
        rows.append(
            {
                "id": case["id"],
                "raw": raw,
                "expected_decision": expected,
                "expected_spans": case["spans"],
                "actual_decision": actual,
                "actual_span": span,
                "exact_decision": exact_decision,
                "scoring_safe": scoring_safe,
                "extractor_status": extraction.get("status"),
                "error": extraction.get("error"),
                "estimated_cost_usd": extraction.get("estimated_cost_usd"),
                "usage": extraction.get("usage"),
            }
        )
    single = [r for r in rows if r["expected_decision"] == "single"]
    negative = [r for r in rows if r["expected_decision"] != "single"]
    summary = {
        "cases": len(rows),
        "exact_decisions": sum(r["exact_decision"] for r in rows),
        "scoring_safe_cases": sum(r["scoring_safe"] for r in rows),
        "single_cases": len(single),
        "single_correct": sum(r["scoring_safe"] for r in single),
        "ambiguous_or_none_cases": len(negative),
        "ambiguous_or_none_safe": sum(r["scoring_safe"] for r in negative),
        "false_single_decisions": sum(r["actual_decision"] == "single" for r in negative),
        "all_scoring_safe": all(r["scoring_safe"] for r in rows),
        "extractor_identity": extractor.identity,
        "estimated_cost_usd": round(sum(float(r["estimated_cost_usd"] or 0) for r in rows), 8),
    }
    return summary, rows


def evaluate_file(
    cases_path: str | Path,
    output_dir: str | Path,
    extractor: AnswerExtractor,
    *,
    offline: bool = False,
) -> dict:
    from wildlife_csi.storage import JsonlExtractionStore

    raw_cases = Path(cases_path).read_bytes()
    cases = [json.loads(line) for line in raw_cases.splitlines() if line.strip()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate extractor case ID")
    cases_hash = hashlib.sha256(raw_cases).hexdigest()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlExtractionStore(
        out / "extractions.jsonl", cases_hash, cases_hash, extractor.identity
    )
    summary, rows = evaluate_cases(cases, extractor, store, offline=offline)
    summary.update(
        {
            "cases_sha256": cases_hash,
            "extractor_config": extractor.public_config,
            "extractions_sha256": hashlib.sha256(store.path.read_bytes()).hexdigest(),
        }
    )
    (out / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
