"""OpenAI-compatible vision transport with complete, safe call records."""

from __future__ import annotations

import base64
import hashlib
import math
import threading
import time
from typing import Any

import httpx

from wildlife_csi.provider_record import (
    ProviderCallError,
    recorded_chat_completion,
    safe_url,
    utc_now,
)


class OpenAnswerAdapter:
    """Send one image and prompt; retain the entire provider response and usage."""

    def __init__(
        self,
        resolved: dict[str, Any],
        client: httpx.Client | None = None,
        now=utc_now,
        monotonic=time.monotonic,
    ):
        self.model_id = resolved["id"]
        self._cfg = resolved
        self._client = client or httpx.Client()
        self._now = now
        self._monotonic = monotonic
        self.totals = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
        }
        self._lock = threading.Lock()
        self._cost_usd = 0.0

    def public_config(self) -> dict[str, Any]:
        cfg = self._cfg
        return {
            "adapter": "openai-compatible",
            "model_id": self.model_id,
            "provider_model": cfg["model"],
            "endpoint": safe_url(cfg["base_url"]),
            "max_output_tokens": cfg["max_output_tokens"],
            "token_param": cfg.get("token_param", "max_tokens"),
            "temperature": cfg.get("temperature"),
            "reasoning_effort": cfg.get("reasoning_effort"),
            "reasoning_style": cfg.get("reasoning_style", "flat"),
            "timeout_s": cfg["timeout_s"],
            "price_per_1k_requests": cfg["price_per_1k_requests"],
            "price_input_1k_tokens": cfg["price_input_1k_tokens"],
            "price_cached_1k_tokens": cfg.get("price_cached_1k_tokens", 0),
            "price_output_1k_tokens": cfg["price_output_1k_tokens"],
            **({"cost_source": "provider"} if cfg.get("cost_source") == "provider" else {}),
        }

    def estimated_cost(self) -> float:
        with self._lock:
            return self._cost_usd

    def provider_cost(self, usage: dict) -> float | None:
        cost = usage.get("cost")
        if (
            self._cfg.get("cost_source") == "provider"
            and type(cost) in (int, float)
            and math.isfinite(cost)
            and cost >= 0
        ):
            return float(cost)
        return None

    def usage_cost(self, usage: dict) -> float:
        if (cost := self.provider_cost(usage)) is not None:
            return cost
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return (
            self._cfg["price_per_1k_requests"] / 1000
            + max(prompt - cached, 0) / 1000 * self._cfg["price_input_1k_tokens"]
            + cached
            / 1000
            * self._cfg.get("price_cached_1k_tokens", self._cfg["price_input_1k_tokens"])
            + completion / 1000 * self._cfg["price_output_1k_tokens"]
        )

    def record_usage(self, usage: dict) -> None:
        with self._lock:
            self.totals["requests"] += 1
            self.totals["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
            self.totals["cached_input_tokens"] += int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            )
            self._cost_usd += self.usage_cost(usage)

    def seed_usage(self, rows: list[dict]) -> None:
        """Include completed rows when a run is resumed under a cost cap."""
        for row in rows:
            if usage := row.get("usage"):
                self.record_usage(usage)

    def predict(
        self, image_bytes: bytes, context: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = context["user_prompt"]
        system_prompt = context.get("system_prompt")
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image_bytes).decode()
                        },
                    },
                ],
            }
        )
        body: dict[str, Any] = {
            "model": self._cfg["model"],
            "messages": messages,
            self._cfg.get("token_param", "max_tokens"): self._cfg["max_output_tokens"],
        }
        if self._cfg.get("temperature") is not None:
            body["temperature"] = self._cfg["temperature"]
        if self._cfg.get("reasoning_effort") and self._cfg.get("reasoning_api", "openai") != "none":
            if self._cfg.get("reasoning_style", "flat") == "nested":
                body["reasoning"] = {"effort": self._cfg["reasoning_effort"]}
            else:
                body["reasoning_effort"] = self._cfg["reasoning_effort"]
        request = {
            "model": body["model"],
            "system_prompt": system_prompt,
            "prompt": prompt,
            "image_sha256": image_sha,
            "image_bytes": len(image_bytes),
            "image_mime": "image/jpeg",
            "parameters": {k: v for k, v in body.items() if k not in ("model", "messages")},
        }
        raw_json, details = recorded_chat_completion(
            base_url=self._cfg["base_url"],
            key_env=self._cfg["key_env"],
            body=body,
            public_request=request,
            timeout_s=self._cfg["timeout_s"],
            client=self._client,
            now=self._now,
            monotonic=self._monotonic,
        )
        try:
            choice = raw_json["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, str):
                answer = content
            elif isinstance(content, list):
                answer = "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                answer = ""
            usage = raw_json.get("usage") or {}
            if not isinstance(usage, dict):
                raise TypeError("usage is not an object")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError(f"unexpected provider payload: {exc}", details) from exc
        self.record_usage(usage)
        return [{"taxon": answer, "score": 1.0}], {
            **details,
            "usage": usage,
            "finish_reason": choice.get("finish_reason"),
            "model": raw_json.get("model", self._cfg["model"]),
            "estimated_cost_usd": self.usage_cost(usage),
            "cost_source": "provider" if self.provider_cost(usage) is not None else "estimate",
        }


def build_adapter(
    model_id: str,
    registry_dir: str = "configs/models",
    client: httpx.Client | None = None,
    factories: dict | None = None,
):
    """Composition root for configured vision providers."""
    from wildlife_csi.registry import load_registry, resolve_model

    reg = load_registry(registry_dir)
    if model_id not in reg:
        raise ValueError(f"unknown model: {model_id} (available: {sorted(reg)})")
    resolved = resolve_model(reg[model_id])
    from wildlife_csi.bedrock_adapter import BedrockAnswerAdapter

    available = {
        "openai-compatible": lambda config: OpenAnswerAdapter(config, client=client),
        "bedrock-converse": lambda config: BedrockAnswerAdapter(config),
    }
    available.update(factories or {})
    kind = resolved["adapter"]
    if kind not in available:
        raise ValueError(f"unsupported adapter type: {kind}")
    return available[kind](resolved)
