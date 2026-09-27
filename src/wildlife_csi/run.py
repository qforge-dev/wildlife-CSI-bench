"""Application wiring for a frozen Wildlife CSI run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wildlife_csi.execution import RunEngine, RunSettings
from wildlife_csi.s3_suite import S3ImageStore
from wildlife_csi.suite import validate_suite
from wildlife_csi.storage import JsonlPredictionStore, LocalImageStore


def run_suite(
    tasks_path: str | Path,
    out_path: str | Path,
    model_id: str,
    images_dir: str | Path | None = None,
    max_cost: float | None = None,
    max_workers: int = 2,
    registry_dir: str = "configs/models",
    *,
    adapter=None,
    image_store=None,
    prediction_store=None,
) -> dict[str, Any]:
    """Validate the frozen suite, then inject provider and filesystem adapters."""
    from wildlife_csi.open_adapter import build_adapter

    tasks, manifest = validate_suite(tasks_path, images_dir)
    if image_store is None:
        if manifest.get("storage", {}).get("kind") == "s3" and images_dir is None:
            image_store = S3ImageStore(region=manifest["storage"]["region"])
        else:
            image_store = LocalImageStore(
                Path(images_dir) if images_dir is not None else Path(tasks_path).parent / "images"
            )
    engine = RunEngine(
        adapter if adapter is not None else build_adapter(model_id, registry_dir),
        image_store,
        prediction_store if prediction_store is not None else JsonlPredictionStore(out_path),
    )
    if engine.adapter.model_id != model_id:
        raise ValueError("injected adapter model ID mismatch")
    return engine.run(tasks, manifest, RunSettings(max_workers=max_workers, max_cost=max_cost))
