"""One model config per YAML file; resolve secrets from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DIR = Path("configs/models")


def load_registry(directory: str | Path = DEFAULT_DIR) -> dict[str, dict[str, Any]]:
    """Load all model configs keyed by id. Raises on duplicate ids."""
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(directory).glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"invalid YAML in {f}: {e}") from e
        mid = data.get("id") or f.stem
        if mid in out:
            raise ValueError(f"duplicate model id: {mid}")
        data["id"] = mid
        data["_file"] = str(f)
        out[mid] = data
    return out


def resolve_model(
    cfg: dict[str, Any], *, require_key: bool = True, require_base: bool = True
) -> dict[str, Any]:
    """Resolve env-backed fields without including secret values."""
    key_env = cfg.get("api_key_env", "")
    adapter = cfg.get("adapter", "openai-compatible")
    cost_source = cfg.get("cost_source", "estimate")
    if cost_source not in ("estimate", "provider"):
        raise ValueError(f"invalid cost_source for model {cfg.get('id')}: {cost_source}")
    if require_key and key_env and adapter != "bedrock-converse" and not os.environ.get(key_env):
        raise RuntimeError(f"missing required env var: {key_env} (model {cfg.get('id')})")
    base_env = cfg.get("base_url_env", "")
    base = os.environ.get(base_env, "") or cfg.get("base_url", "")
    if require_base and adapter == "openai-compatible" and not base:
        raise RuntimeError(f"missing base URL for model {cfg.get('id')} ({base_env or 'base_url'})")
    base_configured = bool(base)
    model = os.environ.get(cfg.get("model_env", ""), "") or cfg.get("model", "")
    if not model:
        raise RuntimeError(f"missing model name for model {cfg.get('id')}")
    return {
        "id": cfg.get("id"),
        "display_name": cfg.get("display_name", cfg.get("id")),
        "adapter": adapter,
        "cost_source": cost_source,
        "base_url": base.rstrip("/"),
        "base_configured": base_configured,
        "base_url_env": base_env,
        "model": model,
        "has_key": bool(key_env and os.environ.get(key_env)),
        "key_env": key_env,
        "max_output_tokens": int(cfg.get("max_output_tokens", 16000)),
        "temperature": (None if cfg.get("temperature") is None else float(cfg.get("temperature"))),
        "token_param": str(cfg.get("token_param", "max_tokens")),
        "reasoning_effort": cfg.get("reasoning_effort"),
        "reasoning_api": str(cfg.get("reasoning_api", "openai")),
        "reasoning_style": str(cfg.get("reasoning_style", "flat")),
        "timeout_s": float(cfg.get("timeout_s", 120)),
        "region": os.environ.get(cfg.get("region_env", ""), "") or cfg.get("region", "us-east-1"),
        "api_version": str(cfg.get("api_version", "") or ""),
        "structured_output": bool(cfg.get("structured_output", False)),
        "price_per_1k_requests": float(cfg.get("price_per_1k_requests", 0.0)),
        "price_input_1k_tokens": float(cfg.get("price_input_1k_tokens", 0.0)),
        "price_cached_1k_tokens": float(
            cfg.get("price_cached_1k_tokens", cfg.get("price_input_1k_tokens", 0.0))
        ),
        "price_output_1k_tokens": float(cfg.get("price_output_1k_tokens", 0.0)),
        "notes": cfg.get("notes", ""),
        "file": cfg.get("_file", ""),
    }


def describe_registry(directory: str | Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    """Safe listing: ids, endpoint host, pricing, key presence. No secrets."""
    rows = []
    for mid, cfg in sorted(load_registry(directory).items()):
        try:
            r = resolve_model(cfg)
        except RuntimeError as e:
            r = {"id": mid, "error": str(e), "has_key": False}
        rows.append(
            {k: v for k, v in r.items() if k != "base_url"}
            | {
                "base_host": (
                    f"aws-bedrock:{r['region']}"
                    if r.get("adapter") == "bedrock-converse"
                    else r.get("base_url", "").split("/")[2]
                    if r.get("base_url")
                    else "?"
                )
            }
        )
    return rows
