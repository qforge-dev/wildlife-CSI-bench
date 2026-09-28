"""Bedrock Converse vision transport with the same durable record contract."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from wildlife_csi.open_adapter import OpenAnswerAdapter
from wildlife_csi.provider_record import ProviderCallError, safe_headers, utc_now


class BedrockAnswerAdapter(OpenAnswerAdapter):
    def __init__(
        self, resolved: dict[str, Any], client=None, now=utc_now, monotonic=time.monotonic
    ):
        super().__init__(resolved, now=now, monotonic=monotonic)
        self._bedrock_client = client

    def _client_for_call(self):
        if self._bedrock_client is None:
            import boto3
            from botocore.config import Config

            key_env = self._cfg.get("key_env")
            if (
                key_env
                and (key := os.environ.get(key_env))
                and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            ):
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self._cfg["region"],
                config=Config(
                    read_timeout=self._cfg["timeout_s"],
                    retries={"total_max_attempts": 1},
                ),
            )
        return self._bedrock_client

    def public_config(self) -> dict[str, Any]:
        return {
            **super().public_config(),
            "adapter": "bedrock-converse",
            "endpoint": f"aws-bedrock:{self._cfg['region']}",
            "region": self._cfg["region"],
        }

    def predict(self, image_bytes: bytes, context: dict[str, Any]):
        prompt = context["user_prompt"]
        system_prompt = context.get("system_prompt")
        request = {
            "model": self._cfg["model"],
            "system_prompt": system_prompt,
            "prompt": prompt,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_bytes": len(image_bytes),
            "image_mime": "image/jpeg",
            "parameters": {
                "maxTokens": self._cfg["max_output_tokens"],
                "reasoning_effort": self._cfg.get("reasoning_effort"),
            },
        }
        started = self._now()
        start = self._monotonic()
        kwargs = {
            "modelId": self._cfg["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": self._cfg["max_output_tokens"]},
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
        if self._cfg.get("reasoning_effort"):
            kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": self._cfg["reasoning_effort"]},
            }
        try:
            response = self._client_for_call().converse(**kwargs)
        except Exception as exc:
            details = {
                "request": request,
                "started_at_utc": started,
                "ended_at_utc": self._now(),
                "api_latency_s": round(self._monotonic() - start, 6),
                "transport_error_type": type(exc).__name__,
                "transport_error": str(exc),
            }
            raise ProviderCallError(f"Bedrock Converse: {exc}", details) from exc
        raw = json.loads(json.dumps(response, default=str))
        metadata = raw.get("ResponseMetadata") or {}
        metadata["HTTPHeaders"] = safe_headers(metadata.get("HTTPHeaders") or {})
        details = {
            "request": request,
            "started_at_utc": started,
            "ended_at_utc": self._now(),
            "api_latency_s": round(self._monotonic() - start, 6),
            "raw_response_json": raw,
            "raw_response_text": json.dumps(raw, ensure_ascii=False),
            "response_headers": metadata["HTTPHeaders"],
            "http_status": metadata.get("HTTPStatusCode"),
        }
        try:
            blocks = raw["output"]["message"]["content"]
            answer = "\n".join(str(b["text"]) for b in blocks if "text" in b)
            bedrock_usage = raw["usage"]
            usage = {
                "prompt_tokens": int(bedrock_usage.get("inputTokens") or 0),
                "completion_tokens": int(bedrock_usage.get("outputTokens") or 0),
                "prompt_tokens_details": {
                    "cached_tokens": int(bedrock_usage.get("cacheReadInputTokens") or 0)
                },
                "provider_usage": bedrock_usage,
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderCallError(f"unexpected Bedrock payload: {exc}", details) from exc
        with self._lock:
            self.totals["requests"] += 1
            self.totals["input_tokens"] += usage["prompt_tokens"]
            self.totals["output_tokens"] += usage["completion_tokens"]
            self.totals["cached_input_tokens"] += usage["prompt_tokens_details"]["cached_tokens"]
        return [{"taxon": answer, "score": 1.0}], {
            **details,
            "usage": usage,
            "finish_reason": raw.get("stopReason"),
            "model": self._cfg["model"],
            "estimated_cost_usd": self.usage_cost(usage),
        }
