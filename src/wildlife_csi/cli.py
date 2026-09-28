"""Wildlife CSI command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)
DEFAULT_SUITE = "data/benchmarks/wildlife-csi-country-v1-2000"
DEFAULT_TASKS = f"{DEFAULT_SUITE}/tasks.jsonl"


def _ensure_local_tasks(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"local suite tasks missing: {path}")


@app.command()
def models(registry: str = "configs/models"):
    import dotenv

    dotenv.load_dotenv(Path(".env"))
    from wildlife_csi.registry import describe_registry

    typer.echo(json.dumps(describe_registry(registry), indent=2))


@app.command()
def validate(
    tasks: str = DEFAULT_TASKS,
    images: str | None = None,
    cache_dir: str = "data/work/csi-s3-cache",
):
    """Validate the frozen metadata and download/check every image anonymously."""
    from wildlife_csi.suite import validate_suite

    _ensure_local_tasks(tasks)
    rows, manifest = validate_suite(tasks, images)
    if manifest.get("storage", {}).get("kind") == "s3" and images is None:
        from concurrent.futures import ThreadPoolExecutor

        from wildlife_csi.s3_suite import S3ImageStore

        store = S3ImageStore(cache=Path(cache_dir), region=manifest["storage"]["region"])
        with ThreadPoolExecutor(max_workers=16) as pool:
            for count, _ in enumerate(pool.map(store.load, rows), 1):
                if count % 250 == 0 or count == len(rows):
                    typer.echo(f"verified {count}/{len(rows)} S3 images", err=True)
    elif images is None:
        validate_suite(tasks, Path(tasks).parent / "images")
    typer.echo(json.dumps({"validated": len(rows), **manifest}, indent=2))


@app.command()
def run(
    tasks: str,
    out: str,
    model: str = "gpt-6-sol",
    images: str | None = None,
    max_cost: float | None = None,
    max_workers: int = 2,
    registry: str = "configs/models",
):
    import dotenv

    dotenv.load_dotenv(Path(".env"))
    from wildlife_csi.run import run_suite

    _ensure_local_tasks(tasks)
    typer.echo(
        json.dumps(
            run_suite(
                tasks,
                out,
                model,
                images,
                max_cost=max_cost,
                max_workers=max_workers,
                registry_dir=registry,
            ),
            indent=2,
        )
    )


@app.command()
def benchmark(
    tasks: str = DEFAULT_TASKS,
    out: str = f"{DEFAULT_SUITE}/runs",
    models: str = typer.Option(..., help="Comma-separated model IDs, or all."),
    max_cost_per_model: float | None = None,
    max_workers: int = 2,
    registry: str = "configs/models",
    extractor_config: str = "configs/extractors/answer.yaml",
):
    """Run the selected models and score their completed predictions."""
    import dotenv

    dotenv.load_dotenv(Path(".env"))
    from wildlife_csi.benchmark import run_benchmark

    _ensure_local_tasks(tasks)
    selected = None if models == "all" else [name.strip() for name in models.split(",")]
    typer.echo(
        json.dumps(
            run_benchmark(
                tasks,
                out,
                selected,
                registry_dir=registry,
                extractor_config=extractor_config,
                max_cost_per_model=max_cost_per_model,
                max_workers=max_workers,
            ),
            indent=2,
        )
    )


@app.command()
def score(
    tasks: str,
    predictions: str,
    offline: bool = False,
    extractor_config: str = "configs/extractors/answer.yaml",
):
    """Score recorded predictions with the official taxonomy and answer extractor."""
    import dotenv

    dotenv.load_dotenv(Path(".env"))
    from wildlife_csi.score import score_run

    _ensure_local_tasks(tasks)
    typer.echo(
        json.dumps(
            score_run(
                tasks,
                predictions,
                offline=offline,
                extractor_config=extractor_config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
