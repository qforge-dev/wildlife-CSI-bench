import hashlib
import json

import httpx
import pytest

from wildlife_csi.bedrock_adapter import BedrockAnswerAdapter
from wildlife_csi.execution import RunEngine, RunSettings
from wildlife_csi.open_adapter import OpenAnswerAdapter, ProviderCallError
from wildlife_csi.scoring import ScoreEngine, align_predictions
from wildlife_csi.source import INaturalist
from wildlife_csi.storage import JsonlPredictionStore, LocalImageStore


class FakeHTTP:
    def __init__(self, response):
        self.response = response
        self.sent = None

    def post(self, url, **kwargs):
        self.sent = {"url": url, **kwargs}
        return self.response


def model_config():
    return {
        "id": "fake-model",
        "model": "deployed-model",
        "base_url": "https://api.example/v1",
        "key_env": "TEST_WILDLIFE_API_KEY",
        "max_output_tokens": 500,
        "token_param": "max_completion_tokens",
        "temperature": None,
        "reasoning_effort": "medium",
        "timeout_s": 30,
        "price_per_1k_requests": 0,
        "price_input_1k_tokens": 0.001,
        "price_cached_1k_tokens": 0.0001,
        "price_output_1k_tokens": 0.002,
    }


def test_adapter_keeps_full_provider_payload_without_key(monkeypatch):
    monkeypatch.setenv("TEST_WILDLIFE_API_KEY", "very-secret")
    response_body = {
        "id": "resp-123",
        "model": "deployed-model",
        "choices": [
            {
                "message": {"content": "The animal is a Killdeer."},
                "finish_reason": "stop",
                "logprobs": {"content": []},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
        "system_fingerprint": "fp-abc",
    }
    response = httpx.Response(
        200,
        json=response_body,
        headers={"x-request-id": "req-789", "set-cookie": "private=1"},
        request=httpx.Request("POST", "https://api.example/v1/chat/completions"),
    )
    client = FakeHTTP(response)
    adapter = OpenAnswerAdapter(model_config(), client=client)
    predictions, info = adapter.predict(
        b"image bytes", {"system_prompt": "Use one name", "user_prompt": "Identify the species"}
    )
    assert predictions[0]["taxon"] == "The animal is a Killdeer."
    assert info["usage"]["completion_tokens_details"]["reasoning_tokens"] == 25
    assert info["raw_response_json"]["system_fingerprint"] == "fp-abc"
    assert info["response_headers"]["x-request-id"] == "req-789"
    assert "set-cookie" not in info["response_headers"]
    assert info["request"]["prompt"] == "Identify the species"
    assert info["request"]["system_prompt"] == "Use one name"
    assert client.sent["json"]["messages"][0] == {"role": "system", "content": "Use one name"}
    assert client.sent["json"]["messages"][1]["role"] == "user"
    assert info["request"]["image_sha256"] == hashlib.sha256(b"image bytes").hexdigest()
    assert "very-secret" not in json.dumps(info)
    assert client.sent["headers"]["Authorization"] == "Bearer very-secret"
    assert info["estimated_cost_usd"] > 0
    assert adapter.estimated_cost() > 0


def test_openrouter_reasoning_uses_nested_parameter(monkeypatch):
    monkeypatch.setenv("TEST_WILDLIFE_API_KEY", "secret")
    response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ANIMAL: fox"}, "finish_reason": "stop"}],
            "usage": {},
        },
        request=httpx.Request("POST", "https://api.example/v1/chat/completions"),
    )
    client = FakeHTTP(response)
    adapter = OpenAnswerAdapter({**model_config(), "reasoning_style": "nested"}, client=client)
    adapter.predict(b"image", {"user_prompt": "Identify the animal"})
    assert client.sent["json"]["reasoning"] == {"effort": "medium"}
    assert "reasoning_effort" not in client.sent["json"]


@pytest.mark.parametrize("charge", [0.0, 0.00001, 0.1])
def test_reported_cost_survives_resume_and_drives_running_total(monkeypatch, charge):
    monkeypatch.setenv("TEST_WILDLIFE_API_KEY", "secret")
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost": charge}
    response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ANIMAL: fox"}, "finish_reason": "stop"}],
            "usage": usage,
        },
        request=httpx.Request("POST", "https://api.example/v1/chat/completions"),
    )
    config = {**model_config(), "cost_source": "provider"}
    adapter = OpenAnswerAdapter(config, client=FakeHTTP(response))
    adapter.seed_usage([{"usage": usage, "estimated_cost_usd": 99}, {"usage": {}}])
    _, info = adapter.predict(b"image", {"user_prompt": "Identify the animal"})
    assert info["estimated_cost_usd"] == charge
    assert info["cost_source"] == "provider"
    assert adapter.estimated_cost() == pytest.approx(2 * charge)


@pytest.mark.parametrize("charge", [None, -1, float("nan"), float("inf"), "0.01", True])
def test_invalid_provider_cost_falls_back_to_token_rates(charge):
    adapter = OpenAnswerAdapter({**model_config(), "cost_source": "provider"})
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost": charge}
    adapter.seed_usage([{"usage": usage}])
    assert adapter.usage_cost(usage) == pytest.approx(0.00014)
    assert adapter.estimated_cost() == pytest.approx(0.00014)


def test_provider_cost_requires_model_opt_in():
    adapter = OpenAnswerAdapter(model_config())
    assert adapter.usage_cost({"prompt_tokens": 100, "cost": 10}) == pytest.approx(0.0001)


def test_bedrock_records_raw_response_and_usage():
    class FakeBedrock:
        def converse(self, **kwargs):
            self.sent = kwargs
            return {
                "output": {"message": {"content": [{"text": "ANIMAL: Vulpes vulpes"}]}},
                "usage": {"inputTokens": 20, "outputTokens": 8, "cacheReadInputTokens": 5},
                "stopReason": "end_turn",
                "ResponseMetadata": {
                    "HTTPStatusCode": 200,
                    "HTTPHeaders": {"x-amzn-requestid": "abc"},
                },
            }

    client = FakeBedrock()
    adapter = BedrockAnswerAdapter(
        {**model_config(), "adapter": "bedrock-converse", "region": "us-east-1"},
        client=client,
    )
    predictions, info = adapter.predict(
        b"image", {"system_prompt": "Use one name", "user_prompt": "Identify the animal"}
    )
    assert predictions[0]["taxon"] == "ANIMAL: Vulpes vulpes"
    assert client.sent["messages"][0]["content"][1]["image"]["source"]["bytes"] == b"image"
    assert client.sent["system"] == [{"text": "Use one name"}]
    assert info["usage"]["prompt_tokens_details"]["cached_tokens"] == 5
    assert info["raw_response_json"]["output"]["message"]["content"][0]["text"]
    assert adapter.estimated_cost() > 0


def test_bedrock_model_key_uses_bearer_env(monkeypatch):
    import boto3
    from botocore.handlers import get_token_from_environment

    monkeypatch.setenv("OPUS55_API_KEY", "test-bedrock-key")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    seen = {}

    def fake_client(*args, **kwargs):
        seen["service"] = args[0]
        seen["token"] = get_token_from_environment("bedrock")
        return object()

    monkeypatch.setattr(boto3, "client", fake_client)
    config = {**model_config(), "region": "us-east-1", "key_env": "OPUS55_API_KEY"}
    BedrockAnswerAdapter(config)._client_for_call()
    assert seen == {"service": "bedrock-runtime", "token": "test-bedrock-key"}


def test_provider_error_keeps_body_and_attempt_details(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_WILDLIFE_API_KEY", "very-secret")
    response = httpx.Response(
        429,
        text='{"error":"rate limited"}',
        headers={"retry-after": "2"},
        request=httpx.Request("POST", "https://api.example/v1/chat/completions"),
    )
    adapter = OpenAnswerAdapter(model_config(), client=FakeHTTP(response))
    try:
        adapter.predict(b"x", {"user_prompt": "Prompt"})
    except ProviderCallError as exc:
        assert exc.details["http_status"] == 429
        assert exc.details["raw_response_text"] == '{"error":"rate limited"}'
        assert exc.details["response_headers"]["retry-after"] == "2"
    else:
        raise AssertionError("expected provider error")


@pytest.mark.parametrize("failure", ["HTTP 429", "Bedrock ServiceUnavailableException"])
def test_run_engine_persists_raw_response_usage_and_retry_history(tmp_path, failure):
    image = b"prepared image"
    sha = hashlib.sha256(image).hexdigest()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / f"{sha}.jpg").write_bytes(image)

    class FlakyAdapter:
        model_id = "fake"
        calls = 0

        def public_config(self):
            return {"adapter": "fake", "model_id": "fake", "temperature": 0.3}

        def seed_usage(self, _rows):
            pass

        def estimated_cost(self):
            return 0.0

        def predict(self, payload, context):
            self.calls += 1
            if self.calls == 1:
                raise ProviderCallError(failure, {"http_status": 429, "raw_response_text": "busy"})
            return (
                [{"taxon": "Killdeer"}],
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "completion_tokens_details": {"reasoning_tokens": 25},
                    },
                    "finish_reason": "stop",
                    "raw_response_json": {"id": "resp-1"},
                    "raw_response_text": '{"id":"resp-1"}',
                    "response_headers": {"x-request-id": "request-1"},
                },
            )

    task = {
        "task_id": "task-1",
        "image_sha256": sha,
        "user_prompt": "What animal?",
        "correct_taxon": "Charadrius vociferus",
    }
    store = JsonlPredictionStore(tmp_path / "run" / "predictions.jsonl")
    engine = RunEngine(FlakyAdapter(), LocalImageStore(image_dir), store, sleep=lambda _: None)
    result = engine.run(
        [task], {"suite": "fixture", "tasks_hash": "hash"}, RunSettings(max_workers=1, retries=1)
    )
    assert result["written"] == 1
    row = store.read()[0]
    assert row["usage"]["completion_tokens_details"]["reasoning_tokens"] == 25
    assert row["provider"]["raw_response_json"]["id"] == "resp-1"
    assert row["attempts_detail"][0]["provider"]["raw_response_text"] == "busy"
    assert row["attempts_detail"][1]["outcome"] == "response"
    assert row["input"]["prompt"] == "What animal?"
    assert store.load_manifest()["sessions"][0]["code_sha256"]


def test_inaturalist_retries_protocol_disconnect(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("disconnected")
        return httpx.Response(200, json={"results": [{"id": 123}]})

    monkeypatch.setattr("wildlife_csi.source.time.sleep", lambda _: None)
    api = INaturalist(httpx.Client(transport=httpx.MockTransport(handle)))
    assert api._get("taxa/123") == [{"id": 123}]
    assert len(calls) == 2


class MemoryCache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, entry):
        self.data[entry["task_id"]] = entry


class UnusedExtractor:
    def extract(self, _raw):
        raise AssertionError("a resolved answer must not call the extractor")


class FakeResolver:
    def resolve_name(self, name):
        assert name == "Killdeer"
        return {"species_id": 1, "genus_id": 2, "family_id": 3}


def test_score_engine_extracts_prose_and_injects_resolver():
    task = {
        "task_id": "t",
        "clue_type": "egg",
        "image_sha256": "a" * 64,
        "correct_taxon": "Charadrius vociferus",
        "correct_taxon_id": 1,
        "correct_genus_id": 2,
        "correct_family_id": 3,
        "location": {"basis": "observation_public_place", "is_observation_location": True},
    }
    pred = {
        "task_id": "t",
        "image_sha256": "a" * 64,
        "tasks_hash": "h",
        "model_id": "fake",
        "status": "answered",
        "predictions": [{"taxon": "The animal is likely a Killdeer. The egg is speckled."}],
    }
    aligned, _ = align_predictions([task], [pred], "h")
    cache = MemoryCache()
    summary, details = ScoreEngine(FakeResolver(), cache, UnusedExtractor(), MemoryCache()).score(
        [task], aligned
    )
    assert summary["overall"] == {"exact": 1}
    assert details[0]["raw_answer"].startswith("The animal")
    assert details[0]["answer"] == "Killdeer"
    assert cache.data["t"]["resolution"]["species_id"] == 1


def test_species_macro_exposes_repeated_species_weighting():
    tasks = [
        {
            "task_id": "a",
            "clue_type": "egg",
            "correct_taxon": "Species alpha",
            "correct_taxon_id": 1,
            "correct_genus_id": 10,
            "correct_family_id": 100,
            "image_sha256": "a" * 64,
        },
        {
            "task_id": "b",
            "clue_type": "egg",
            "correct_taxon": "Species alpha",
            "correct_taxon_id": 1,
            "correct_genus_id": 10,
            "correct_family_id": 100,
            "image_sha256": "b" * 64,
        },
        {
            "task_id": "c",
            "clue_type": "egg",
            "correct_taxon": "Species beta",
            "correct_taxon_id": 2,
            "correct_genus_id": 20,
            "correct_family_id": 200,
            "image_sha256": "c" * 64,
        },
    ]
    for task in tasks:
        task["location"] = {"basis": "observation_public_place", "is_observation_location": True}
    predictions = {
        task["task_id"]: {
            "status": "answered",
            "predictions": [{"taxon": "Species alpha" if task["task_id"] != "c" else "Uria aalge"}],
        }
        for task in tasks
    }

    class WrongResolver:
        def resolve_name(self, name):
            return {"species_id": 99, "genus_id": 98, "family_id": 97}

    result, _ = ScoreEngine(WrongResolver(), MemoryCache(), UnusedExtractor(), MemoryCache()).score(
        tasks, predictions
    )
    assert result["exact_accuracy"] == 2 / 3
    assert result["species_macro"]["exact_accuracy"] == 0.5
    assert result["species_macro"]["species"] == 2
    assert result["coverage"] == 1


def test_coverage_distinguishes_abstention_from_wrong_answer():
    tasks = [
        {
            "task_id": "a",
            "clue_type": "egg",
            "correct_taxon": "Species alpha",
            "correct_taxon_id": 1,
            "correct_genus_id": 10,
            "correct_family_id": 100,
            "image_sha256": "a" * 64,
        },
        {
            "task_id": "b",
            "clue_type": "egg",
            "correct_taxon": "Species beta",
            "correct_taxon_id": 2,
            "correct_genus_id": 20,
            "correct_family_id": 200,
            "image_sha256": "b" * 64,
        },
    ]
    for task in tasks:
        task["location"] = {"basis": "observation_public_place", "is_observation_location": True}
    predictions = {
        "a": {"status": "answered", "predictions": [{"taxon": "Species alpha"}]},
        "b": {"status": "answered", "predictions": [{"taxon": "UNKNOWN"}]},
    }
    tasks[1]["location"] = {"basis": "species_occurrence_example", "is_observation_location": False}
    result, _ = ScoreEngine(FakeResolver(), MemoryCache(), UnusedExtractor(), MemoryCache()).score(
        tasks, predictions
    )
    assert result["overall"] == {"exact": 1, "abstain": 1}
    assert result["coverage"] == 0.5
    assert result["exact_accuracy"] == 0.5
    assert result["exact_accuracy_answered"] == 1.0
    assert result["per_location_basis"]["species_occurrence_example"]["tasks"] == 1
    assert result["per_location_basis"]["observation_public_place"]["exact_accuracy"] == 1.0
