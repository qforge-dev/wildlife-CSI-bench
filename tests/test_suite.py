import hashlib
import io
import json

import pytest
import httpx
from PIL import Image

from wildlife_csi.animalclue import (
    AccessPending,
    crop_box,
    observation_id,
    parse_yolo_label,
    require_access,
)
from wildlife_csi.parse import SYSTEM_PROMPT, parse_answer, user_prompt
from wildlife_csi.suite import (
    _validate_task,
    build_task,
    prepare_image,
    validate_suite,
    write_manifest,
)
from wildlife_csi.run import run_suite
from wildlife_csi.score import score_run
from wildlife_csi.scoring import extract_answer
from wildlife_csi.source import INaturalist, photo_reference, source_record


def sample_truth():
    return {
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
    }


def sample_location():
    return {
        "country": "United States",
        "level": "country",
        "basis": "observation_public_place",
        "is_observation_location": True,
        "source": "iNaturalist",
        "source_place_id": 1,
        "geoprivacy": "open",
        "observation_id": "123",
    }


def test_selected_dataset_access_fails_before_build(monkeypatch):
    monkeypatch.setattr(
        "wildlife_csi.animalclue.check_access",
        lambda: {"egg": {"bytes": "ok"}, "bone": {"bytes": "pending-author-review"}},
    )
    require_access(["egg"])
    with pytest.raises(AccessPending, match="bone.*pending-author-review"):
        require_access(["egg", "bone"])


def sample_suite(tmp_path):
    image = b"test image bytes"
    sha = hashlib.sha256(image).hexdigest()
    images = tmp_path / "images"
    images.mkdir()
    (images / f"{sha}.jpg").write_bytes(image)
    task = build_task(
        "egg",
        sha,
        {
            "repo": "fixture",
            "revision": "abc",
            "observation_id": "123",
            "image_path": "123_0.jpg",
            "inat": sample_truth(),
        },
        sample_location(),
    )
    write_manifest(tmp_path, [task], 42)
    return task, sha


def test_source_taxonomy_overrides_class_mapping():
    task = build_task(
        "egg",
        "a" * 64,
        {
            "observation_id": "123",
            "image_path": "123_0.jpg",
            "yolo_class_id": 7,
            "inat": sample_truth(),
        },
        sample_location(),
    )
    assert task["correct_taxon"] == "Vulpes vulpes"
    assert task["correct_family"] == "Canidae"
    assert photo_reference("species/test/images/123_0_jpeg.rf.abc.jpg") == ("123", 0)
    assert observation_id("123_0_jpeg.rf.abc.jpg") == "123"


def test_official_protocol_rejects_other_prompts_and_versions():
    task = build_task(
        "egg",
        "a" * 64,
        {"observation_id": "123", "image_path": "123_0.jpg", "inat": sample_truth()},
        sample_location(),
    )
    with pytest.raises(ValueError, match="protocol or prompt"):
        _validate_task({**task, "protocol_version": 11})
    with pytest.raises(ValueError, match="protocol or prompt"):
        _validate_task({**task, "user_prompt": "A different question"})


def test_prompt_does_not_reveal_location_provenance():
    observed = sample_location()
    estimated = {**observed, "basis": "species_occurrence_example", "is_observation_location": False}
    assert user_prompt(observed) == user_prompt(estimated)
    assert "private" not in user_prompt(estimated)


def test_source_record_checks_observation_photo_and_rank():
    lineage = [
        {"id": 3, "name": "Canidae", "rank": "family"},
        {"id": 2, "name": "Vulpes", "rank": "genus"},
    ]
    taxon = {"id": 1, "name": "Vulpes vulpes", "rank": "species", "ancestors": lineage}
    obs = {
        "id": 123,
        "quality_grade": "research",
        "taxon": {"id": 1},
        "photos": [{"id": 9, "license_code": "cc-by", "attribution": "X"}],
    }
    assert source_record(obs, taxon, "123_0.jpg")["species"] == "Vulpes vulpes"
    with pytest.raises(ValueError, match="photo index"):
        source_record(obs, taxon, "123_1.jpg")
    with pytest.raises(ValueError, match="not research"):
        source_record({**obs, "quality_grade": "needs_id"}, taxon, "123_0.jpg")


def test_inaturalist_throttling_cools_down_then_retries():
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "normal_throttling"})
        return httpx.Response(200, json={"results": [{"id": 42}]})

    api = INaturalist(httpx.Client(transport=httpx.MockTransport(respond)))
    cooldowns = []
    api._pace_request = lambda: None
    api._cooldown = cooldowns.append
    assert api._get("taxa/42") == [{"id": 42}]
    assert len(calls) == 2
    assert cooldowns == [60.0]


def test_manifest_and_image_validation(tmp_path):
    task, sha = sample_suite(tmp_path)
    assert validate_suite(tmp_path / "tasks.jsonl", tmp_path / "images")[0] == [task]
    (tmp_path / "images" / f"{sha}.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        validate_suite(tmp_path / "tasks.jsonl", tmp_path / "images")
    with pytest.raises(FileExistsError):
        write_manifest(tmp_path, [task], 42)


def test_source_snapshot_tamper_is_rejected(tmp_path):
    sample_suite(tmp_path)
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text(source_path.read_text().replace("Vulpes vulpes", "Vulpes lagopus"))
    with pytest.raises(ValueError, match="source snapshot hash"):
        validate_suite(tmp_path / "tasks.jsonl")


def test_location_snapshot_and_wrong_protocol_are_rejected(tmp_path):
    task, _ = sample_suite(tmp_path)
    locations = tmp_path / "locations.jsonl"
    locations.write_text(locations.read_text().replace("United States", "Canada"))
    with pytest.raises(ValueError, match="location snapshot hash"):
        validate_suite(tmp_path / "tasks.jsonl")
    locations.write_text(json.dumps({"task_id": task["task_id"], "location": task["location"]}) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["suite"] = "wildlife-csi-v1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="suite or protocol"):
        validate_suite(tmp_path / "tasks.jsonl")


def test_selection_ledger_is_validated(tmp_path):
    task, _ = sample_suite(tmp_path)
    ledger = (json.dumps({"task_id": task["task_id"], "selected": True}) + "\n").encode()
    (tmp_path / "selection.jsonl").write_bytes(ledger)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection"] = {"ledger_sha256": hashlib.sha256(ledger).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    validate_suite(tmp_path / "tasks.jsonl")
    (tmp_path / "selection.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="selection ledger hash"):
        validate_suite(tmp_path / "tasks.jsonl")


def test_score_requires_matching_hash_and_ignores_stale_verdict(tmp_path):
    task, sha = sample_suite(tmp_path)
    pred = {
        "task_id": task["task_id"],
        "image_sha256": sha,
        "tasks_hash": "old",
        "model_id": "fixture",
        "status": "answered",
        "predictions": [{"taxon": "Vulpes vulpes"}],
        "verdict": {"level": "none", "ground_truth": "wrong truth"},
    }
    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text(json.dumps(pred) + "\n")
    with pytest.raises(ValueError, match="task hash"):
        score_run(tmp_path / "tasks.jsonl", pred_path)
    pred["image_sha256"] = "0" * 64
    pred_path.write_text(json.dumps(pred) + "\n")
    with pytest.raises(ValueError, match="image mismatch"):
        score_run(tmp_path / "tasks.jsonl", pred_path)
    pred["image_sha256"] = sha
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    pred["tasks_hash"] = manifest["tasks_hash"]
    pred_path.write_text(json.dumps(pred) + "\n")
    result = score_run(tmp_path / "tasks.jsonl", pred_path, offline=True)
    assert result["overall"] == {"exact": 1}
    assert result["exact_accuracy"] == 1


def test_injected_adapter_run_resumes_from_recorded_prediction(tmp_path):
    task, _ = sample_suite(tmp_path)
    output = tmp_path / "reference.jsonl"

    class FixedAdapter:
        model_id = "fixture"

        def public_config(self):
            return {"adapter": "fixture", "model_id": self.model_id}

        def predict(self, _image, _context):
            return [{"taxon": "Vulpes vulpes", "score": 1.0}], {"usage": {}}

        def seed_usage(self, _rows):
            pass

        def estimated_cost(self):
            return 0.0

    adapter = FixedAdapter()
    result = run_suite(
        tmp_path / "tasks.jsonl", output, "fixture", images_dir=tmp_path / "images",
        adapter=adapter,
    )
    assert result["written"] == 1
    row = json.loads(output.read_text())
    assert row["predictions"][0]["taxon"] == task["correct_taxon"]
    assert (
        run_suite(
            tmp_path / "tasks.jsonl", output, "fixture", images_dir=tmp_path / "images",
            adapter=adapter,
        )["already_done"]
        == 1
    )


def test_image_and_parser():
    img = Image.new("RGB", (640, 480), "brown")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    payload, meta = prepare_image(buf.getvalue())
    assert payload[:2] == b"\xff\xd8" and meta["width"] == 640
    assert parse_answer("Vulpes vulpes")["answer"] == "Vulpes vulpes"
    assert parse_answer("a\nb")["status"] == "invalid"
    assert parse_answer("ANIMAL: Blue-and-yellow macaw")["answer"] == "Blue-and-yellow macaw"
    assert parse_answer("ANIMAL: Little tern (Sternula albifrons)") == {
        "status": "answered",
        "trace": "field_binomial",
        "answer": "Sternula albifrons",
    }
    assert parse_answer("ANIMAL: Wild boar (*Sus scrofa*)") == {
        "status": "answered",
        "trace": "field_binomial",
        "answer": "Sus scrofa",
    }
    assert parse_answer("ANIMAL: Sus scrofa (wild boar)") == {
        "status": "answered",
        "trace": "field_binomial",
        "answer": "Sus scrofa",
    }
    assert (
        parse_answer("ANIMAL: Little tern (Sternula albifrons) or Killdeer")["status"] == "invalid"
    )
    assert parse_answer("ANIMAL: fox or wolf")["status"] == "invalid"
    assert "United States" in user_prompt(sample_location())
    assert "ANIMAL: <name>" in SYSTEM_PROMPT
    assert parse_yolo_label("3 0.5 0.5 0.2 0.2\n")[0][0] == 3
    assert crop_box(1000, 800, 0.5, 0.5, 0.2, 0.2)[2] > 500


def test_resolver_accepts_exact_matched_common_name_and_species_over_subspecies():
    api = INaturalist()
    species = {
        "id": 482,
        "name": "Fulica atra",
        "rank": "species",
        "ancestors": [
            {"id": 100, "rank": "family"},
            {"id": 200, "rank": "genus"},
        ],
    }
    lookups = {
        "taxa/autocomplete?q=Common%20blackbird": [
            {
                "id": 12716,
                "name": "Turdus merula",
                "rank": "species",
                "matched_term": "Common Blackbird",
            }
        ],
        "taxa/12716": [
            {
                "id": 12716,
                "name": "Turdus merula",
                "rank": "species",
                "ancestors": [
                    {"id": 300, "rank": "family"},
                    {"id": 400, "rank": "genus"},
                ],
            }
        ],
        "taxa/autocomplete?q=Eurasian%20coot": [
            {"id": 482, "name": "Fulica atra", "rank": "species", "matched_term": "Eurasian Coot"},
            {
                "id": 337558,
                "name": "Fulica atra atra",
                "rank": "subspecies",
                "matched_term": "Eurasian coot",
                "ancestor_ids": [482],
            },
        ],
        "taxa/482": [species],
        "taxa/autocomplete?q=Sceloporus%20undulatus": [
            {"id": 1690776, "name": "Sceloporus undulatus", "rank": "complex"},
            {"id": 36142, "name": "Sceloporus undulatus", "rank": "species",
             "ancestor_ids": [1690776, 36142]},
        ],
        "taxa/36142": [
            {"id": 36142, "name": "Sceloporus undulatus", "rank": "species",
             "ancestors": [{"id": 999, "rank": "family"},
                           {"id": 36141, "rank": "genus"},
                           {"id": 1690776, "rank": "complex"}]}
        ],
        "taxa/autocomplete?q=Pica%20pica": [
            {"id": 891696, "name": "Pica pica", "rank": "species", "matched_term": "Pica pica"},
            {
                "id": 286502,
                "name": "Urera baccifera",
                "rank": "species",
                "matched_term": "pica pica",
            },
        ],
        "taxa/891696": [
            {
                "id": 891696,
                "name": "Pica pica",
                "rank": "species",
                "ancestors": [{"id": 301, "rank": "family"}, {"id": 401, "rank": "genus"}],
            }
        ],
        "taxa/autocomplete?q=Rock%20pigeon": [
            {
                "id": 3017,
                "name": "Columba livia",
                "rank": "species",
                "preferred_common_name": "Rock Pigeon",
                "matched_term": "Rock Pigeon",
            },
            {
                "id": 3036,
                "name": "Columba guinea",
                "rank": "species",
                "matched_term": "Rock Pigeon",
            },
        ],
        "taxa/3017": [
            {
                "id": 3017,
                "name": "Columba livia",
                "rank": "species",
                "ancestors": [{"id": 302, "rank": "family"}, {"id": 402, "rank": "genus"}],
            }
        ],
    }
    api._get = lookups.__getitem__
    assert api.resolve_name("Common blackbird")["species_id"] == 12716
    assert api.resolve_name("Eurasian coot")["species_id"] == 482
    assert api.resolve_name("Sceloporus undulatus")["species_id"] == 36142
    assert api.resolve_name("Pica pica")["species_id"] == 891696
    assert api.resolve_name("Rock pigeon")["species_id"] == 3017


@pytest.mark.parametrize(
    ("response", "answer"),
    [
        ("The animal is likely a Killdeer. The eggs are speckled.", "Killdeer"),
        ("I think this is a Killdeer (Charadrius vociferus).", "Charadrius vociferus"),
        ("I think the bird is a Killdeer because of the egg markings.", "Killdeer"),
        ("**Killdeer** — the eggs look speckled.", "Killdeer"),
        ("Killdeer. The speckled eggs fit.", "Killdeer"),
        ('{"species": "Charadrius vociferus", "confidence": 0.7}', "Charadrius vociferus"),
        ("Species: Red fox\nReason: the color pattern", "Red fox"),
    ],
)
def test_parser_extracts_one_name_from_wrapped_response(response, answer):
    assert parse_answer(response)["answer"] == answer


def test_parser_rejects_multiple_names():
    assert (
        parse_answer("It might be Charadrius vociferus or Vanellus vanellus")["status"]
        == "ambiguous"
    )


def test_final_animal_field_takes_precedence_over_explanation():
    reply = "Maybe Vulpes vulpes or Canis lupus.\nANIMAL: gray wolf"
    parsed = parse_answer(reply)
    assert parsed["status"] == "answered"
    assert parsed["answer"] == "gray wolf"
    assert parsed["trace"] == "field"
    assert parse_answer("ANIMAL: red fox\nANIMAL: gray wolf")["status"] == "ambiguous"
    assert parse_answer("ANIMAL: red fox or gray wolf")["status"] == "invalid"
    assert extract_answer("ANIMAL: UNKNOWN")["status"] == "abstain"
