import hashlib
import json

import httpx
import pytest
from helpers import build_task, write_manifest

from wildlife_csi.answer_extractor import TextAnswerExtractor, validate_decision
from wildlife_csi.provider_record import ProviderCallError
from wildlife_csi.score import score_run
from wildlife_csi.scoring import ScoreEngine
from wildlife_csi.storage import JsonlExtractionStore, JsonlResolutionStore


def test_extractor_accepts_hyphenated_animal_name():
    raw = "ANIMAL: Blue-and-yellow macaw"
    decision = {
        "decision": "single",
        "source_span": "Blue-and-yellow macaw",
        "uncertain": False,
        "reason": "one named animal",
    }
    assert validate_decision(decision, raw)["source_span"] == "Blue-and-yellow macaw"


def task():
    return {
        "task_id": "t",
        "clue_type": "egg",
        "image_sha256": "a" * 64,
        "correct_taxon": "Vulpes vulpes",
        "correct_taxon_id": 1,
        "correct_genus_id": 2,
        "correct_family_id": 3,
        "location": {"basis": "observation_public_place", "is_observation_location": True},
    }


def prediction(raw):
    return {"task_id": "t", "status": "answered", "predictions": [{"taxon": raw}]}


class Resolver:
    def __init__(self):
        self.names = []

    def resolve_name(self, name):
        self.names.append(name)
        return {"species_id": 1, "genus_id": 2, "family_id": 3} if name == "red fox" else None


class FakeExtractor:
    identity = "fake-v1"
    public_config = {"model": "fake", "prompt_version": "fake-v1"}

    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def extract(self, raw):
        self.calls.append(raw)
        return {
            "status": "ok",
            "decision": self.decision,
            "usage": {"prompt_tokens": 25, "completion_tokens": 10},
            "estimated_cost_usd": 0.001,
            "provider": {"raw_response_text": "full text", "api_latency_s": 0.5},
        }


def stores(tmp_path):
    return (
        JsonlResolutionStore(tmp_path / "resolutions.jsonl", "p" * 64, "suite"),
        JsonlExtractionStore(tmp_path / "extractions.jsonl", "p" * 64, "suite", "fake-v1"),
    )


def test_extractor_only_for_unclear_reply_and_offline_cache(tmp_path):
    raw = "I would identify this as a red fox, from the shell markings."
    decision = {
        "decision": "single",
        "source_span": "red fox",
        "uncertain": False,
        "reason": "one final animal",
    }
    extractor = FakeExtractor(decision)
    resolver = Resolver()
    cache, extractor_cache = stores(tmp_path)
    engine = ScoreEngine(resolver, cache, extractor, extractor_cache)
    summary, details = engine.score([task()], {"t": prediction(raw)})
    assert summary["overall"] == {"exact": 1}
    assert summary["extractor_attempted"] == summary["extractor_assisted"] == 1
    assert summary["provisional_due_to_extractor"] is True
    assert details[0]["match_method"] == "extractor_assisted_taxonomy"
    assert details[0]["review_required"] is True
    assert extractor.calls == [raw]
    assert resolver.names == ["red fox"]
    stored = json.loads((tmp_path / "extractions.jsonl").read_text())
    assert stored["raw_answer_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert stored["extraction"]["provider"]["raw_response_text"] == "full text"
    assert stored["extraction"]["usage"]["prompt_tokens"] == 25
    replay = ScoreEngine(Resolver(), cache, FakeExtractor(decision), extractor_cache)
    replay_summary, _ = replay.score([task()], {"t": prediction(raw)}, offline=True)
    assert replay_summary["overall"] == summary["overall"]
    assert replay.extractor.calls == []


def test_clear_answer_skips_extractor_and_unsubstantiated_decision_gets_no_credit(tmp_path):
    decision = {
        "decision": "single",
        "source_span": "Red fox",
        "uncertain": False,
        "reason": "one name",
    }
    cache, extractor_cache = stores(tmp_path)
    extractor = FakeExtractor(decision)
    engine = ScoreEngine(Resolver(), cache, extractor, extractor_cache)
    summary, _ = engine.score([task()], {"t": prediction("Vulpes vulpes")}, offline=True)
    assert summary["overall"] == {"exact": 1}
    assert summary["extractor_attempted"] == 0
    assert extractor.calls == []
    unclear = "The shell is speckled, but I cannot name an animal."
    summary, details = engine.score([task()], {"t": prediction(unclear)})
    assert summary["extractor_attempted"] == 1
    assert summary["extractor_assisted"] == 0
    assert details[0]["review_required"] is True
    assert details[0]["level"] != "exact"


def test_resolved_wrong_animal_is_scored_without_extractor(tmp_path):
    class WrongTaxonomy:
        def resolve_name(self, name):
            assert name == "Gray wolf"
            return {"species_id": 99, "genus_id": 98, "family_id": 97}

    cache, extractor_cache = stores(tmp_path)
    extractor = FakeExtractor(
        {
            "decision": "none",
            "source_span": "",
            "uncertain": False,
            "reason": "",
        }
    )
    summary, details = ScoreEngine(WrongTaxonomy(), cache, extractor, extractor_cache).score(
        [task()], {"t": prediction("Gray wolf")}
    )
    assert summary["overall"] == {"wrong": 1}
    assert summary["extractor_attempted"] == 0
    assert extractor.calls == []
    assert details[0]["match_method"] == "taxonomy_id"


def test_text_extractor_sends_only_reply_and_records_entire_provider_call(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    raw = "I would identify this as a Red fox, from the shell markings."
    decision = {
        "decision": "single",
        "source_span": "Red fox",
        "uncertain": False,
        "reason": "one final animal",
    }
    body = {
        "id": "extractor-1",
        "model": "extractor-model",
        "choices": [{"message": {"content": json.dumps(decision)}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 55,
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }

    class Client:
        def post(self, url, **kwargs):
            self.sent = {"url": url, **kwargs}
            return httpx.Response(
                200,
                json=body,
                headers={"x-request-id": "request-1"},
                request=httpx.Request("POST", url),
            )

    client = Client()
    config = {
        "model": "extractor-model",
        "base_url": "https://example.test/v1",
        "key_env": "TEST_JUDGE_KEY",
        "max_output_tokens": 1000,
        "token_param": "max_tokens",
        "reasoning_effort": "medium",
        "temperature": None,
        "timeout_s": 10,
        "price_per_1k_requests": 0,
        "price_input_1k_tokens": 0.001,
        "price_cached_1k_tokens": 0.0001,
        "price_output_1k_tokens": 0.002,
    }
    result = TextAnswerExtractor(config, client=client).extract(raw)
    assert result["decision"] == decision
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4
    assert result["provider"]["raw_response_json"]["id"] == "extractor-1"
    assert result["provider"]["response_headers"]["x-request-id"] == "request-1"
    sent = json.dumps(client.sent["json"])
    assert raw in sent and "Vulpes vulpes" not in sent and "correct_taxon" not in sent
    assert "secret" not in json.dumps(result)


def test_invalid_extractor_span_is_rejected():
    with pytest.raises(ValueError, match="source span"):
        validate_decision(
            {
                "decision": "single",
                "source_span": "Red fox",
                "uncertain": False,
                "reason": "guess",
            },
            "No animal named",
        )


def test_extractor_name_cannot_receive_canonical_credit_without_taxonomy(tmp_path):
    raw = "Vulpes vulpes or maybe something else."
    decision = {
        "decision": "single",
        "source_span": "Vulpes vulpes",
        "uncertain": True,
        "reason": "one animal named",
    }

    class UnresolvedTaxonomy:
        def resolve_name(self, name):
            return None

    cache, extractor_cache = stores(tmp_path)
    engine = ScoreEngine(UnresolvedTaxonomy(), cache, FakeExtractor(decision), extractor_cache)
    summary, details = engine.score([task()], {"t": prediction(raw)})
    assert summary["extractor_attempted"] == 1
    assert summary["extractor_assisted"] == 0
    assert details[0]["level"] == "ambiguous"


def test_extractor_cannot_replace_quoted_name_with_different_normalization(tmp_path):
    raw = "My tentative answer from the tracks is a gray wolf, given their size."
    decision = {
        "decision": "single",
        "source_span": "gray wolf",
        "normalized_name": "Vulpes vulpes",
        "uncertain": True,
        "reason": "one named animal",
    }

    class GrayWolfTaxonomy:
        def resolve_name(self, name):
            assert name == "gray wolf"
            return {"species_id": 99, "genus_id": 98, "family_id": 97}

    cache, extractor_cache = stores(tmp_path)
    summary, details = ScoreEngine(
        GrayWolfTaxonomy(), cache, FakeExtractor(decision), extractor_cache
    ).score([task()], {"t": prediction(raw)})
    assert summary["overall"] == {"wrong": 1}
    assert details[0]["answer"] == "gray wolf"


def test_provider_error_is_incomplete_and_retried_with_full_history(tmp_path):
    raw = "I would identify this as a red fox, from the shell markings."
    decision = {
        "decision": "single",
        "source_span": "red fox",
        "uncertain": False,
        "reason": "one animal",
    }

    class FlakyExtractor(FakeExtractor):
        def extract(self, answer):
            self.calls.append(answer)
            if len(self.calls) == 1:
                raise ProviderCallError(
                    "HTTP 429", {"http_status": 429, "raw_response_text": "rate limited"}
                )
            return {
                "status": "ok",
                "decision": self.decision,
                "provider": {"raw_response_text": "valid"},
                "usage": {},
            }

    cache, extractor_cache = stores(tmp_path)
    extractor = FlakyExtractor(decision)
    engine = ScoreEngine(Resolver(), cache, extractor, extractor_cache)
    first, _ = engine.score([task()], {"t": prediction(raw)})
    assert first["score_complete"] is False
    assert first["extractor_provider_errors"] == 1
    second, _ = engine.score([task()], {"t": prediction(raw)})
    assert second["score_complete"] is True
    assert second["overall"] == {"exact": 1}
    events = [
        json.loads(line) for line in (tmp_path / "extractions.jsonl").read_text().splitlines()
    ]
    assert [e["extraction"]["status"] for e in events] == ["provider_error", "ok"]


def test_score_run_writes_separate_extractor_audit_and_review_queue(tmp_path):
    image = b"image"
    sha = hashlib.sha256(image).hexdigest()
    images = tmp_path / "images"
    images.mkdir()
    (images / f"{sha}.jpg").write_bytes(image)
    source = {
        "repo": "fixture",
        "revision": "abc",
        "observation_id": "123",
        "image_path": "123_0.jpg",
        "inat": {
            "observation_id": "123",
            "species": "Vulpes vulpes",
            "species_id": 1,
            "genus": "Vulpes",
            "genus_id": 2,
            "family": "Canidae",
            "family_id": 3,
            "photo_id": 9,
            "photo_index": 0,
            "photo_license": "cc-by",
        },
    }
    suite_task = build_task(
        "egg",
        sha,
        source,
        {
            "country": "United States",
            "level": "country",
            "basis": "observation_public_place",
            "is_observation_location": True,
            "source": "iNaturalist",
            "source_place_id": 1,
            "geoprivacy": "open",
            "observation_id": "123",
        },
    )
    write_manifest(tmp_path, [suite_task], 42)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    raw = "I would identify this as a red fox, from the shell markings."
    row = {
        "task_id": suite_task["task_id"],
        "image_sha256": sha,
        "tasks_hash": manifest["tasks_hash"],
        "model_id": "fixture",
        "status": "answered",
        "predictions": [{"taxon": raw}],
    }
    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text(json.dumps(row) + "\n")
    decision = {
        "decision": "single",
        "source_span": "red fox",
        "uncertain": False,
        "reason": "one final animal",
    }
    extractor = FakeExtractor(decision)
    result = score_run(
        tmp_path / "tasks.jsonl", pred_path, resolver=Resolver(), extractor=extractor
    )
    assert result["overall"] == {"exact": 1}
    out = tmp_path / "score"
    assert (
        result["extractions_sha256"]
        == hashlib.sha256((out / "extractions.jsonl").read_bytes()).hexdigest()
    )
    assert json.loads((out / "review_queue.jsonl").read_text())["review_required"] is True
    assert (
        json.loads((out / "scores.jsonl").read_text())["match_method"]
        == "extractor_assisted_taxonomy"
    )
    replay = FakeExtractor(decision)
    second = score_run(
        tmp_path / "tasks.jsonl", pred_path, resolver=Resolver(), extractor=replay, offline=True
    )
    assert second["overall"] == result["overall"]
    assert replay.calls == []
