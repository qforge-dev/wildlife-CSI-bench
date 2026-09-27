# Wildlife CSI

Wildlife CSI tests whether a general-purpose vision model can identify the animal that left a footprint, dropping, egg, bone, or feather. Each task supplies one vetted image and its country, with no candidate list. The frozen suite has 2,000 tasks: 400 per trace type.

The official suite is `data/benchmarks/wildlife-csi-country-v1-2000/`. Its task hash is `b1d0ee38219c2f63542ef6d39249ab2735b976287bb775101f5eba73cd877b42`. Do not edit its tasks, prompts, source snapshots, or location records after running a model. A changed dataset or prompt needs a new suite identity and fresh runs.

## Prompt and scoring

System message:

> Identify animal traces from photos. Give one best species guess, preferably a scientific name. Reply on one line: ANIMAL: <name>. No explanation or alternatives.

User message, with the country filled from each task:

> I found this animal trace in the field. The photo's country is United States. Which species most likely left it?

The scorer parses a named answer, checks it against iNaturalist taxonomy IDs, and awards cumulative species, genus, and family credit. A text-only extractor handles answers that the parser cannot resolve; it sees the model reply but neither the image nor the correct answer. Its calls and decisions are recorded for review. Every task remains in the denominator, including provider errors and unanswered replies. See [current results](reports/results.md) and the [country provenance audit](reports/location-audit.md).

## Run a model

From the repository root:

```bash
uv sync --extra dev
test -f .env || cp .env.example .env
uv run csi models
uv run csi validate
uv run csi benchmark --models deepseek --max-workers 10
```

Fill `.env` with the endpoint and key for the models you select. `csi models` lists configured models without printing secrets. Each model has one YAML file in `configs/models/`; all current evaluation configs request high reasoning effort. `csi benchmark` preflights credentials, resumes recorded image calls when necessary, and scores complete runs. For more control, use `csi run <tasks.jsonl> <predictions.jsonl> --model <id> --max-workers 10`, followed by `csi score <tasks.jsonl> <predictions.jsonl>`. The command help shows all options.

The canonical layout is `runs/<model>/predictions.jsonl`, `run_manifest.json`, and `score/` within the suite directory. Prediction records preserve the prompt, adapter settings, complete provider response, usage, timing, retries, errors, and estimated cost. Score files preserve every grade, taxon resolution, extractor call, and review item. Their hashes bind them to the frozen task and prediction snapshots. `csi score --offline` reproduces a score when its recorded resolution and extraction caches are complete.

The private bucket `wildlife-csi-dataset-088543363904` holds only JPEG image objects under `media/sha256/`. Task metadata and run records stay local. Image reads verify SHA-256 and use the disposable `data/work/csi-s3-cache/` cache.

## Dataset provenance

Images come from AnimalClue's test splits. The task records retain pinned source revisions, iNaturalist observation taxonomy, photo attribution, and a selection audit. The source metadata has no country field; country comes from each source observation's public iNaturalist place records. Four private observations have no public country and use a documented species-range example instead. Their task rows explicitly mark `location.is_observation_location: false` and retain a source URL. The same country prompt is used for all 2,000 tasks.

Some source photos have noncommercial or missing licenses. Review the recorded photo rights and trace diagnosability before publishing a leaderboard or redistributing images.
