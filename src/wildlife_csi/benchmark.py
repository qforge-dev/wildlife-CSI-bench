"""Run and score each configured model against the same frozen suite."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from wildlife_csi.registry import load_registry, resolve_model
from wildlife_csi.run import run_suite
from wildlife_csi.score import score_run
from wildlife_csi.suite import validate_suite


def run_benchmark(
    tasks_path: str | Path,
    out_dir: str | Path,
    model_ids: list[str] | None = None,
    *,
    registry_dir: str | Path = "configs/models",
    max_cost_per_model: float | None = None,
    max_workers: int = 2,
    extractor_config: str | Path = "configs/extractors/answer.yaml",
) -> dict:
    """Fail credential preflight before calls; resume each model in its own folder."""
    registry = load_registry(registry_dir)
    selected = model_ids if model_ids is not None else list(registry)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("select at least one distinct model")
    unknown = set(selected) - registry.keys()
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if max_cost_per_model is not None and max_cost_per_model < 0:
        raise ValueError("max_cost_per_model must be nonnegative")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    missing = []
    for model_id in selected:
        cfg = registry[model_id]
        if cfg.get("adapter", "openai-compatible") not in ("openai-compatible", "bedrock-converse"):
            raise ValueError(f"unsupported adapter for {model_id}: {cfg.get('adapter')}")
        try:
            resolved = resolve_model(cfg)
        except RuntimeError as exc:
            missing.append(str(exc))
            continue
        if cfg.get("adapter") == "bedrock-converse" and not (
            resolved["has_key"] or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        ):
            try:
                import boto3

                if boto3.Session().get_credentials() is None:
                    missing.append(f"missing AWS credentials for {model_id}")
            except Exception as exc:
                missing.append(f"AWS credential check failed for {model_id}: {type(exc).__name__}")
    if missing:
        raise RuntimeError("model preflight failed:\n" + "\n".join(missing))
    extractor_cfg = yaml.safe_load(Path(extractor_config).read_text())
    resolve_model(extractor_cfg)
    tasks, manifest = validate_suite(tasks_path)
    output = Path(out_dir)
    results = {}
    for model_id in selected:
        predictions = output / model_id / "predictions.jsonl"
        run = run_suite(
            tasks_path,
            predictions,
            model_id,
            max_cost=max_cost_per_model,
            max_workers=max_workers,
            registry_dir=str(registry_dir),
        )
        result = {"run": run, "predictions": str(predictions)}
        if run["already_done"] + run["written"] == len(tasks):
            result["score"] = score_run(tasks_path, predictions, extractor_config=extractor_config)
        else:
            result["score"] = None
            result["incomplete"] = True
        results[model_id] = result
    return {"suite": manifest["suite"], "tasks_hash": manifest["tasks_hash"], "models": results}
