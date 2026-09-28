import hashlib
import io

import pytest

from wildlife_csi.s3_suite import S3ImageStore


class FakeS3:
    def __init__(self, payload=b"image bytes"):
        self.payload = payload
        self.gets = 0

    def get_object(self, **_):
        self.gets += 1
        return {"Body": io.BytesIO(self.payload)}


def test_s3_image_store_verifies_remote_bytes_and_cache(tmp_path):
    payload = b"image bytes"
    sha = hashlib.sha256(payload).hexdigest()
    task = {
        "image_sha256": sha,
        "image_s3_uri": f"s3://private-bucket/media/sha256/{sha[:2]}/{sha}.jpg",
    }
    client = FakeS3(payload)
    store = S3ImageStore(cache=tmp_path, client=client)
    assert store.load(task) == payload
    assert store.load(task) == payload
    assert client.gets == 1
    (tmp_path / sha[:2] / sha).write_bytes(b"corrupted cache")
    assert store.load(task) == payload
    assert client.gets == 2
    client.payload = b"wrong remote image"
    (tmp_path / sha[:2] / sha).write_bytes(b"corrupted cache")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load(task)


def test_download_client_never_loads_aws_credentials(monkeypatch):
    import botocore.session
    from botocore import UNSIGNED

    from wildlife_csi.s3_suite import s3_client

    def unexpected_credentials(*args, **kwargs):
        raise AssertionError("public images must not request AWS credentials")

    monkeypatch.setattr(botocore.session.Session, "get_credentials", unexpected_credentials)
    client = s3_client()
    assert client.meta.config.signature_version == UNSIGNED
    client.close()


def test_invalid_checksum_cannot_be_used_as_cache_path(tmp_path):
    client = FakeS3()
    store = S3ImageStore(cache=tmp_path, client=client)
    with pytest.raises(ValueError, match="invalid image SHA-256"):
        store.load({"image_sha256": "../../outside-cache", "image_s3_uri": "s3://bucket/image"})
    assert client.gets == 0
    assert not list(tmp_path.iterdir())
