"""Pure taxonomic grading plus injected answer-resolution service."""

from __future__ import annotations

import re
from collections import Counter

from wildlife_csi.answer_extractor import answer_sha256, validate_decision
from wildlife_csi.parse import parse_answer
from wildlife_csi.ports import AnswerExtractor, ExtractionStore, ResolutionStore, TaxonResolver
from wildlife_csi.provider_record import ProviderCallError

SCORER_VERSION = "wildlife-csi-country-scorer"


def _species_macro(groups: dict[int, Counter]) -> dict:
    """Average each species' accuracy so repeated images do not dominate."""
    return {
        "species": len(groups),
        "exact_accuracy": sum(c["exact"] / c["total"] for c in groups.values()) / len(groups),
        "genus_accuracy": sum((c["exact"] + c["genus_only"]) / c["total"] for c in groups.values())
        / len(groups),
        "family_accuracy": sum(
            (c["exact"] + c["genus_only"] + c["family_only"]) / c["total"] for c in groups.values()
        )
        / len(groups),
    }


def raw_answer(row: dict) -> str:
    return str((row.get("predictions") or [{}])[0].get("taxon") or "")


def extract_answer(raw: str) -> dict:
    if re.match(r"^\s*UNKNOWN(?:\s*$|\s*[,;:.—-]|\s+because\b)", raw, re.IGNORECASE):
        return {"status": "abstain", "trace": "abstain", "answer": "UNKNOWN"}
    parsed = parse_answer(raw)
    if parsed["status"] == "answered" and parsed["answer"].casefold() == "unknown":
        return {"status": "abstain", "trace": parsed["trace"], "answer": "UNKNOWN"}
    return parsed


def grade(task: dict, answer: dict, taxon: dict | None, *, allow_canonical: bool = True) -> str:
    if answer["status"] != "answered":
        return answer["status"]
    if allow_canonical and " ".join(answer["answer"].casefold().split()) == " ".join(
        task["correct_taxon"].casefold().split()
    ):
        return "exact"
    if taxon is None:
        return "unresolved"
    if taxon.get("species_id") == task["correct_taxon_id"]:
        return "exact"
    if taxon.get("genus_id") == task["correct_genus_id"]:
        return "genus_only"
    if taxon.get("family_id") == task["correct_family_id"]:
        return "family_only"
    return "wrong"


def align_predictions(
    tasks: list[dict], rows: list[dict], tasks_hash: str
) -> tuple[dict[str, dict], str]:
    """Reject predictions from any other official task snapshot."""
    by_task = {task["task_id"]: task for task in tasks}
    by_image = {task["image_sha256"]: task for task in tasks}
    if len(by_task) != len(tasks) or len(by_image) != len(tasks):
        raise ValueError("duplicate suite task/image")
    predictions: dict[str, dict] = {}
    hashes, model_ids = set(), set()
    for row in rows:
        tid = row["task_id"]
        if tid not in by_task or tid in predictions:
            raise ValueError("prediction has unknown or duplicate task ID")
        if row.get("image_sha256") != by_task[tid]["image_sha256"]:
            raise ValueError(f"prediction image mismatch: {tid}")
        if not row.get("tasks_hash"):
            raise ValueError("prediction missing tasks_hash")
        hashes.add(row["tasks_hash"])
        model_ids.add(row.get("model_id"))
        predictions[tid] = row
    if len(hashes) != 1 or len(model_ids) != 1 or None in model_ids:
        raise ValueError("predictions contain mixed/missing task hashes or model IDs")
    run_hash = hashes.pop()
    if run_hash != tasks_hash:
        raise ValueError("prediction task hash differs from official suite")
    return predictions, model_ids.pop()


class ScoreEngine:
    """Resolve answer names through injected taxonomy and cache ports."""

    def __init__(
        self,
        resolver: TaxonResolver,
        cache: ResolutionStore,
        extractor: AnswerExtractor,
        extractor_store: ExtractionStore,
    ):
        self.resolver = resolver
        self.cache = cache
        self.extractor = extractor
        self.extractor_store = extractor_store

    def _extract_unresolved(
        self, task_id: str, raw: str, *, offline: bool
    ) -> tuple[dict, dict | None]:
        """Persist the costly call before resolving its extracted name."""
        cached = self.extractor_store.get(task_id)
        if cached is not None and cached["raw_answer_sha256"] != answer_sha256(raw):
            raise ValueError("extractor cache answer mismatch")
        if cached is None or (
            cached["extraction"].get("status") == "provider_error" and not offline
        ):
            if offline:
                raise ValueError(f"extractor answer missing from cache: {task_id}")
            try:
                extraction = self.extractor.extract(raw)
            except ProviderCallError as exc:
                extraction = {
                    "status": "provider_error",
                    "error": str(exc),
                    "provider": exc.details,
                }
            cached = {
                "task_id": task_id,
                "raw_answer_sha256": answer_sha256(raw),
                "extraction": extraction,
            }
            self.extractor_store.put(cached)
        extraction = cached["extraction"]
        decision = extraction.get("decision") or {}
        if extraction.get("status") == "ok":
            try:
                decision = validate_decision(decision, raw)
            except ValueError:
                return cached, None
        if extraction.get("status") != "ok" or decision.get("decision") != "single":
            return cached, None
        name = decision["source_span"]
        resolution_key = task_id + ":extractor"
        resolution = self.cache.get(resolution_key)
        if resolution is not None and resolution["answer"] != name:
            raise ValueError("extractor resolution cache answer mismatch")
        if resolution is None:
            if offline:
                raise ValueError(f"extractor resolution missing from cache: {task_id}")
            resolution = {
                "task_id": resolution_key,
                "answer": name,
                "resolution": self.resolver.resolve_name(name),
            }
            self.cache.put(resolution)
        return cached, resolution["resolution"]

    def score(
        self,
        tasks: list[dict],
        predictions: dict[str, dict],
        *,
        offline: bool = False,
    ) -> tuple[dict, list[dict]]:
        counts = Counter()
        per_type: dict[str, Counter] = {}
        per_location_basis: dict[str, Counter] = {}
        answered_count = 0
        answered_by_type: Counter = Counter()
        by_species: dict[int, Counter] = {}
        by_type_species: dict[str, dict[int, Counter]] = {}
        details = []
        resolved_names: dict[str, dict | None] = {}
        extractor_attempted = 0
        extractor_assisted = 0
        extractor_provider_errors = 0
        extractor_cost = 0.0
        extractor_usage: Counter = Counter()
        parse_methods: Counter = Counter()
        model_answered_rows = 0
        for task in tasks:
            pred = predictions.get(task["task_id"])
            raw = raw_answer(pred) if pred else ""
            answer = (
                extract_answer(raw)
                if pred and pred.get("status") == "answered"
                else {
                    "status": "missing" if pred is None else pred.get("status", "invalid"),
                    "answer": "",
                }
            )
            parsed_answer = answer.copy()
            if pred and pred.get("status") == "answered":
                model_answered_rows += 1
                parse_methods[parsed_answer.get("trace") or parsed_answer["status"]] += 1
            cached = self.cache.get(task["task_id"])
            if cached and cached["answer"] != answer["answer"]:
                raise ValueError("resolution cache answer mismatch")
            needs_resolution = answer["status"] == "answered" and " ".join(
                answer["answer"].casefold().split()
            ) != " ".join(task["correct_taxon"].casefold().split())
            if needs_resolution and cached is None:
                if offline:
                    raise ValueError(f"answer missing from resolution cache: {task['task_id']}")
                name = answer["answer"]
                if name not in resolved_names:
                    resolved_names[name] = self.resolver.resolve_name(name)
                cached = {
                    "task_id": task["task_id"],
                    "answer": name,
                    "resolution": resolved_names[name],
                }
                self.cache.put(cached)
            resolution = cached["resolution"] if cached else None
            level = grade(task, answer, resolution)
            extraction_record = None
            assisted = False
            if (
                raw.strip()
                and pred
                and pred.get("status") == "answered"
                and (
                    answer["status"] in {"invalid", "ambiguous"}
                    or (answer["status"] == "answered" and resolution is None and needs_resolution)
                )
            ):
                extraction_record, extracted_resolution = self._extract_unresolved(
                    task["task_id"], raw, offline=offline
                )
                extractor_attempted += 1
                extraction = extraction_record["extraction"]
                if extraction.get("status") == "provider_error":
                    extractor_provider_errors += 1
                extractor_cost += float(extraction.get("estimated_cost_usd") or 0)
                usage = extraction.get("usage") or {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    extractor_usage[key] += int(usage.get(key) or 0)
                decision = extraction.get("decision") or {}
                if decision.get("decision") == "single" and extracted_resolution is not None:
                    answer = {"status": "answered", "answer": decision["source_span"]}
                    level = grade(task, answer, extracted_resolution, allow_canonical=False)
                    resolution = extracted_resolution
                    assisted = True
                    extractor_assisted += 1
            counts[level] += 1
            per_type.setdefault(task["clue_type"], Counter())[level] += 1
            location = task["location"]
            per_location_basis.setdefault(location["basis"], Counter())[level] += 1
            species_id = task["correct_taxon_id"]
            species_counts = by_species.setdefault(species_id, Counter())
            type_species_counts = by_type_species.setdefault(task["clue_type"], {}).setdefault(
                species_id, Counter()
            )
            for species_counter in (species_counts, type_species_counts):
                species_counter[level] += 1
                species_counter["total"] += 1
            if answer["status"] == "answered":
                answered_count += 1
                answered_by_type[task["clue_type"]] += 1
            details.append(
                {
                    "task_id": task["task_id"],
                    "clue_type": task["clue_type"],
                    "location_basis": location["basis"],
                    "is_observation_location": location["is_observation_location"],
                    "image_sha256": task["image_sha256"],
                    "truth": task["correct_taxon"],
                    "truth_species_id": task["correct_taxon_id"],
                    "truth_genus_id": task["correct_genus_id"],
                    "truth_family_id": task["correct_family_id"],
                    "raw_answer": raw,
                    "parsed_answer": parsed_answer["answer"],
                    "answer": answer["answer"],
                    "parse_status": parsed_answer["status"],
                    "parse_method": parsed_answer.get("trace"),
                    "level": level,
                    "match_method": (
                        "extractor_assisted_taxonomy"
                        if assisted
                        else "canonical_name"
                        if level == "exact" and resolution is None
                        else "taxonomy_id"
                        if resolution is not None
                        else None
                    ),
                    "resolution": resolution,
                    "extractor_attempted": extraction_record is not None,
                    "extractor_assisted": assisted,
                    "review_required": extraction_record is not None,
                    "extractor_decision": (
                        extraction_record["extraction"].get("decision")
                        if extraction_record
                        else None
                    ),
                    "extractor_status": (
                        extraction_record["extraction"].get("status") if extraction_record else None
                    ),
                    "extractor_raw_answer_sha256": (
                        extraction_record["raw_answer_sha256"] if extraction_record else None
                    ),
                }
            )
        n = len(tasks)
        return (
            {
                "scorer_version": SCORER_VERSION,
                "extractor_attempted": extractor_attempted,
                "extractor_assisted": extractor_assisted,
                "extractor_provider_errors": extractor_provider_errors,
                "score_complete": extractor_provider_errors == 0,
                "review_required": extractor_attempted,
                "provisional_due_to_extractor": extractor_assisted > 0,
                "extractor_estimated_cost_usd": round(extractor_cost, 8),
                "extractor_usage": dict(extractor_usage),
                "parse_methods": dict(parse_methods),
                "field_compliance": (
                    (parse_methods["field"] + parse_methods["field_binomial"]) / model_answered_rows
                    if model_answered_rows
                    else None
                ),
                "overall": dict(counts),
                "per_type": {k: dict(v) for k, v in per_type.items()},
                "per_location_basis": {
                    basis: {
                        "tasks": sum(basis_counts.values()),
                        "exact_accuracy": basis_counts["exact"] / sum(basis_counts.values()),
                        "genus_accuracy": (
                            basis_counts["exact"] + basis_counts["genus_only"]
                        ) / sum(basis_counts.values()),
                        "family_accuracy": (
                            basis_counts["exact"] + basis_counts["genus_only"]
                            + basis_counts["family_only"]
                        ) / sum(basis_counts.values()),
                    }
                    for basis, basis_counts in per_location_basis.items()
                },
                "answered": answered_count,
                "coverage": answered_count / n,
                "per_type_coverage": {
                    clue: answered_by_type[clue] / sum(c.values()) for clue, c in per_type.items()
                },
                "exact_accuracy": counts["exact"] / n,
                "genus_accuracy": (counts["exact"] + counts["genus_only"]) / n,
                "family_accuracy": (counts["exact"] + counts["genus_only"] + counts["family_only"])
                / n,
                "exact_accuracy_answered": (
                    counts["exact"] / answered_count if answered_count else None
                ),
                "species_macro": _species_macro(by_species),
                "per_type_species_macro": {
                    clue: _species_macro(groups) for clue, groups in by_type_species.items()
                },
            },
            details,
        )
