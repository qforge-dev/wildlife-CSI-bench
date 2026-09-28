import json
from pathlib import Path

import httpx
import pytest
import yaml
from helpers import build_task, write_manifest
from typer.testing import CliRunner

from wildlife_csi import benchmark
from wildlife_csi.cli import app
from wildlife_csi.open_adapter import build_adapter


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.chdir(tmp_path)
    image = b"synthetic image"
    sha = hashlib.sha256(image).hexdigest()
    images = tmp_path / "images"
    images.mkdir()
    (images / f"{sha}.jpg").write_bytes(image)
    task = build_task(
        "egg",
        sha,
        {
            "observation_id": "123",
            "image_path": "123_0.jpg",
            "inat": {
                "observation_id": "123",
                "photo_index": 0,
                "species": "Vulpes vulpes",
                "species_id": 1,
                "genus": "Vulpes",
                "genus_id": 2,
                "family": "Canidae",
                "family_id": 3,
            },
        },
        {
            "country": "United States",
            "level": "country",
            "observation_id": "123",
            "basis": "observation_public_place",
            "is_observation_location": True,
        },
    )
    write_manifest(tmp_path, [task], 42)
    registry = tmp_path / "models"
    registry.mkdir()
    config = {
        "id": "custom",
        "model": "custom-model",
        "adapter": "openai-compatible",
        "api_key_env": "CSI_TEST_KEY",
        "base_url": "https://provider.example/v1",
    }
    (registry / "custom.yaml").write_text(yaml.safe_dump(config))
    extractor = tmp_path / "extractor.yaml"
    extractor.write_text(yaml.safe_dump(config))
    # Exercise the CLI's .env loader without relying on the developer's credentials.
    (tmp_path / ".env").write_text("CSI_TEST_KEY=test-only-key\n")
    monkeypatch.delenv("CSI_TEST_KEY", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    args = [
        "benchmark",
        "--tasks",
        str(tmp_path / "tasks.jsonl"),
        "--out",
        str(tmp_path / "runs"),
        "--registry",
        str(registry),
        "--models",
        "custom",
        "--extractor-config",
        str(extractor),
    ]
    return args, config, extractor


def test_custom_model_runs_scores_and_resumes_from_dotenv(setup, monkeypatch):
    args, _, extractor = setup
    calls = []

    def respond(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-only-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ANIMAL: Vulpes vulpes"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        "wildlife_csi.open_adapter.build_adapter",
        lambda model, registry: build_adapter(model, registry, client=client),
    )
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    result_data = json.loads(result.output)["models"]["custom"]
    assert result_data["run"]["written"] == 1
    assert result_data["score"]["exact_accuracy"] == 1.0
    assert result_data["score"]["score_complete"]
    assert len(calls) == 1
    assert "test-only-key" not in Path(result_data["predictions"]).read_text()

    resumed = runner.invoke(app, args)
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.output)["models"]["custom"]["run"]["already_done"] == 1
    assert len(calls) == 1

    offline = runner.invoke(
        app,
        [
            "score",
            "tasks.jsonl",
            result_data["predictions"],
            "--offline",
            "--extractor-config",
            str(extractor),
        ],
    )
    assert offline.exit_code == 0, offline.output
    assert json.loads(offline.output)["exact_accuracy"] == 1.0
    assert len(calls) == 1


def test_benchmark_requires_explicit_model_selection():
    result = CliRunner().invoke(app, ["benchmark"])
    assert result.exit_code == 2
    assert "--models" in result.output


@pytest.mark.parametrize("key_env", ["OPUS55_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"])
def test_bedrock_bearer_preflight_does_not_require_aws_access_keys(setup, monkeypatch, key_env):
    args, config, extractor = setup
    config.update(adapter="bedrock-converse", api_key_env="OPUS55_API_KEY")
    Path("models/custom.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv(key_env, "test-bearer")

    def unexpected_session():
        raise AssertionError("bearer authentication must not require AWS access keys")

    monkeypatch.setattr("boto3.Session", unexpected_session)
    monkeypatch.setattr(benchmark, "run_suite", lambda *a, **kw: {"already_done": 0, "written": 0})
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["models"]["custom"]["incomplete"]
