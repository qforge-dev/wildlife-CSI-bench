"""Anonymous, checksum-verified reads of the published benchmark images."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config

DEFAULT_REGION = "us-east-1"
DEFAULT_CACHE = Path("data/work/csi-s3-cache")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def s3_client(region: str = DEFAULT_REGION):
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            signature_version=UNSIGNED,
            retries={"mode": "standard", "max_attempts": 4},
            max_pool_connections=16,
        ),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        handle.write(data)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class S3ImageStore:
    """Read task image bytes through a disposable SHA-256 verified cache."""

    def __init__(self, cache: Path = DEFAULT_CACHE, client=None, region: str = DEFAULT_REGION):
        self.cache = Path(cache)
        self.client = client or s3_client(region)

    def load(self, task: dict) -> bytes:
        sha = task["image_sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid image SHA-256")
        path = self.cache / sha[:2] / sha
        if path.exists():
            data = path.read_bytes()
            if digest(data) == sha:
                return data
        uri = urlparse(task["image_s3_uri"])
        if uri.scheme != "s3" or not uri.netloc or not uri.path.lstrip("/"):
            raise ValueError("task has invalid S3 image URI")
        body = self.client.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        if digest(data) != sha:
            raise ValueError(f"S3 image checksum mismatch: {uri.geturl()}")
        _atomic_write(path, data)
        return data
