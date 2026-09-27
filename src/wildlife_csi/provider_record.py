"""Shared recorded HTTP transport for vision calls and text-only extractors."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def safe_headers(headers: Any) -> dict[str, str]:
    blocked = {
        "set-cookie",
        "cookie",
        "authorization",
        "proxy-authorization",
        "www-authenticate",
        "proxy-authenticate",
    }
    return {str(k): str(v) for k, v in headers.items() if str(k).lower() not in blocked}


class ProviderCallError(RuntimeError):
    """Provider failure carrying the sanitized request and full response."""

    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


def recorded_chat_completion(
    *,
    base_url: str,
    key_env: str,
    body: dict,
    public_request: dict,
    timeout_s: float,
    client: httpx.Client,
    now: Callable[[], str] = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[dict, dict]:
    """POST a chat completion and return raw JSON plus durable call metadata.

    Authorization and image base64 remain out of the saved request record.
    The request hash covers the canonical full JSON body so it can be checked
    against the separately stored image bytes and prompt later.
    """
    key = os.environ.get(key_env, "")
    if not key:
        raise RuntimeError(f"missing API key: {key_env}")
    url = base_url.rstrip("/") + "/chat/completions"
    request = {
        **public_request,
        "method": "POST",
        "endpoint": safe_url(url),
        "request_json_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    started_at = now()
    start = monotonic()
    try:
        response = client.post(
            url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout_s
        )
    except Exception as exc:
        details = {
            "request": request,
            "started_at_utc": started_at,
            "ended_at_utc": now(),
            "api_latency_s": round(monotonic() - start, 6),
            "transport_error_type": type(exc).__name__,
            "transport_error": str(exc),
        }
        raise ProviderCallError(f"{type(exc).__name__}: {exc}", details) from exc
    raw_text = response.text
    try:
        raw_json = response.json()
    except ValueError:
        raw_json = None
    details = {
        "request": request,
        "started_at_utc": started_at,
        "ended_at_utc": now(),
        "api_latency_s": round(monotonic() - start, 6),
        "http_status": response.status_code,
        "response_headers": safe_headers(response.headers),
        "raw_response_text": raw_text,
        "raw_response_json": raw_json,
    }
    if response.status_code >= 400:
        raise ProviderCallError(f"HTTP {response.status_code}", details)
    if not isinstance(raw_json, dict):
        raise ProviderCallError("provider returned non-object JSON", details)
    return raw_json, details
