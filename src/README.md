# Running Wildlife CSI

[Description and results](../README.md)

Use **Python 3.12+** and run commands from the **repository root**. The repository includes the frozen task metadata. Images download automatically from S3 using anonymous requests; **no AWS account or Hugging Face login is needed for the dataset**.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/qforge-dev/wildlife-CSI-bench.git
cd wildlife-CSI-bench
uv sync
test -f .env || cp .env.example .env
uv run csi models
```

Alternatively, use a virtual environment and pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
csi models
```

With pip, omit `uv run` from the commands below. The CLI loads `.env` from the repository root; existing shell variables take precedence. Keep credentials in `.env`, which Git ignores.

## Configure a model

Edit [.env.example](../.env.example)'s copied values in `.env`. You only need credentials for the selected model and the answer extractor used for scoring.

| Model ID | Required credentials | Optional overrides |
| --- | --- | --- |
| `gpt-6-sol` | `GPT6_SOL_API_KEY`, `GPT6_SOL_BASE_URL` | `GPT6_SOL_MODEL` |
| `astra` | `GPT6_ASTRA_API_KEY`, `GPT6_ASTRA_BASE_URL` | `GPT6_ASTRA_MODEL` |
| `luna` | `GPT6_LUNA_API_KEY`, `GPT6_LUNA_BASE_URL` | `GPT6_LUNA_MODEL` |
| `deepseek` | `DEEPSEEK_API_KEY` (OpenRouter) | `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` |
| `glm` | `GLM_API_KEY` (OpenRouter) | `GLM_BASE_URL`, `GLM_MODEL` |
| `muse` | `MUSE_API_KEY` (OpenRouter) | `MUSE_BASE_URL`, `MUSE_MODEL` |
| `grok` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL`, `GROK_MODEL` |
| `gemini` | `GEMINI_API_KEY` | `GEMINI_BASE_URL`, `GEMINI_MODEL` |
| `opus` | `OPUS55_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, or standard AWS credentials | `OPUS55_REGION`, `OPUS55_MODEL` |
| `sonnet` | `SONNET55_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, or standard AWS credentials | `SONNET55_REGION`, `SONNET55_MODEL` |
| `fable` | `AWS_BEARER_TOKEN_BEDROCK` or standard AWS credentials | `FABLE_REGION`, `FABLE_MODEL` |

For OpenAI, use `https://api.openai.com/v1` as the base URL. For Azure, use `https://YOUR-RESOURCE.openai.azure.com/openai/v1` and set the model override to your deployment name. Bedrock authentication applies only to model calls; image downloads stay anonymous. Muse's Contributor tier requires accepting the provider's data-use terms.

The default [answer extractor](../configs/extractors/answer.yaml) uses `GPT6_SOL_BASE_URL`, `GPT6_SOL_API_KEY`, and `GPT6_SOL_MODEL`. Configure those even when benchmarking another model. It only reads model replies the parser cannot resolve. To use your own compatible text model, copy that YAML, change its model and environment-variable fields, and pass `--extractor-config path/to/answer.yaml` to `csi benchmark` or `csi score`. Changing the extractor can affect comparability with the published scores.

`uv run csi models` lists model IDs and missing configuration without printing keys. Each model's settings, reasoning effort, token limit, and estimated prices live in [configs/models](../configs/models/). Verify rates for your provider before a paid run.

OpenRouter configs set `cost_source: provider`: recorded USD charges from `usage.cost` take precedence over token-rate estimates, including when enforcing a spending limit or resuming a run. Missing or invalid charges fall back to the YAML rates. The `estimated_cost_usd` output field keeps its existing name; `provider.cost_source` identifies reported charges versus estimates. AWS and Azure costs remain token-rate estimates. Published raw runs retain their original estimates; the charts use their saved OpenRouter charges where available.

## Download and validate the images

```bash
uv run csi validate
```

This checks task, source, location, and selection hashes, then downloads and verifies all 2,000 images. Subsequent runs reuse `data/work/csi-s3-cache/`. Corrupt cached images are downloaded again; a remote checksum mismatch stops the run.

## Run and score

For a complete run with one model:

```bash
uv run csi benchmark --models deepseek --max-workers 10
```

Select multiple models with `--models deepseek,glm`. `--models all` explicitly selects every configured model and requires their credentials. The runner checks credentials before making model calls and scores each model once all predictions are recorded.

To check your setup with a spending limit:

```bash
uv run csi benchmark --models deepseek --max-workers 2 --max-cost-per-model 1
```

The limit stops new requests after recorded estimated spending reaches $1; in-flight calls can exceed it. A limited run can be incomplete and will not be automatically scored. Repeat the command with a higher limit, or without the limit, to finish it.

Outputs default to `data/benchmarks/wildlife-csi-country-v1-2000/runs/<model>/`. Repeating the same command resumes recorded predictions. Use a new `--out`, such as `--out data/work/my-experiment`, for a fresh experiment or changed model configuration. Provider errors already recorded as predictions remain part of that run.

### Separate generation and scoring

Generation needs only the selected vision model's credentials:

```bash
uv run csi run \
  data/benchmarks/wildlife-csi-country-v1-2000/tasks.jsonl \
  data/work/my-run/predictions.jsonl \
  --model deepseek --max-workers 10

uv run csi score \
  data/benchmarks/wildlife-csi-country-v1-2000/tasks.jsonl \
  data/work/my-run/predictions.jsonl
```

Add `--offline` to `csi score` once the run's taxonomy-resolution and extraction caches are complete. Keep the same extractor configuration and endpoint when replaying its cache. The scorer rejects caches belonging to different task, prediction, or extractor snapshots.

Use `uv run csi --help` or a command's `--help` for all options. `csi run --images /path/to/images` accepts a local directory containing `<image_sha256>.jpg` files instead of downloading from S3.

## Add your own model

Copy a YAML from [configs/models](../configs/models/) to a new file in that directory. Give it a unique `id`, set the provider's `model` name, and point `api_key_env` and `base_url_env` at variables in your `.env`. Adjust token limits, reasoning options, and prices for that provider. Then select the new ID with `--models your-model`.

The runner supports OpenAI-compatible chat-completion vision endpoints and Bedrock Converse. It records the resolved settings with each run. Keep the frozen photos and prompts unchanged when comparing against the published results.

## Outputs

| File | Contents |
| --- | --- |
| `predictions.jsonl` | Model answers, prompts, complete provider responses, usage, timing, retries, and estimated costs |
| `run_manifest.json` | Suite identity, model settings, and execution sessions |
| `score/summary.json` | Species, genus, and family scores, per-trace results, failures, and provenance hashes |
| `score/scores.jsonl` | Per-photo grades and how each answer was resolved |
| `score/resolutions.jsonl` | Recorded taxonomy resolutions for replay |
| `score/extractions.jsonl` | Text-extractor calls and decisions |
| `score/review_queue.jsonl` | Answers requiring manual review |

Your generated runs, caches, and credentials are excluded from Git. The seven published runs live separately in [runs](../runs/), with full predictions compressed as `predictions.jsonl.gz`. Each `publication.json` records original and published file hashes. Private Azure resource hostnames are replaced with `azure-endpoint.invalid`; model answers, usage, costs, and grades are preserved.

## Replay the published runs

The [run index](../runs/index.json) links each published result to its evidence. Predictions are gzip-compressed to keep the repository manageable and stay below GitHub's file-size limit. Manifests, summaries, and per-photo scores are ordinary JSON/JSONL files.

To reproduce Opus's score without provider credentials or network calls:

```bash
mkdir -p data/work/replay-opus
gzip -dc runs/opus/predictions.jsonl.gz > data/work/replay-opus/predictions.jsonl
cp -R runs/opus/score data/work/replay-opus/score
uv run csi score \
  data/benchmarks/wildlife-csi-country-v1-2000/tasks.jsonl \
  data/work/replay-opus/predictions.jsonl \
  --offline --extractor-config runs/opus/extractor.yaml
```

Replace `opus` with another directory from `runs/` to replay that model. Use the run's `extractor.yaml`: it preserves the published cache identity and uses a non-routable endpoint. It is only for offline replay; new model runs use your own provider configuration.

The chart snapshot retains the original prediction hashes. Each run's `publication.json` maps those hashes to the published copies after endpoint redaction and cache rebinding. Recorded per-photo scores are unchanged. Local logs, the taxonomy database, and the interrupted Opus outage attempt are not part of the published results.

## Protocol and scoring

There are 400 photos each of bones, eggs, feathers, droppings, and footprints. Every photo uses this system message:

```text
Identify animal traces from photos. Give one best species guess, preferably a scientific name. Reply on one line: ANIMAL: <name>. No explanation or alternatives.
```

The user message varies only by country:

```text
I found this animal trace in the field. The photo's country is South Africa. Which species most likely left it?
```

The parser extracts one animal name and the scorer resolves it against iNaturalist taxonomy IDs. Credit is cumulative: species matches also count toward genus and family. The text extractor sees only the model's reply. Failed calls, empty replies, and invalid answers remain in the denominator. Inspect `score_complete`, `provisional_due_to_extractor`, and the review queue before comparing scores.

The official [suite](../data/benchmarks/wildlife-csi-country-v1-2000/) has task hash:

```text
b1d0ee38219c2f63542ef6d39249ab2735b976287bb775101f5eba73cd877b42
```

Tasks, prompts, source snapshots, location records, and the selection ledger are frozen. Changes to them require a new suite identity and fresh runs. Image access settings in the manifest describe hosting and do not change the task hash.

Countries come from public iNaturalist place records. Four private observations use documented species-range examples and are marked `location.is_observation_location: false`; these are answer-derived hints. One public-place country conflicts with free-text locality, and one Puerto Rico observation is represented as United States. The scorer reports results by location basis.

Images were normalized to metadata-free RGB JPEGs, with a minimum short side of 320 pixels, a maximum long side of 1,536 pixels, and a 3 MB size cap. Source revisions, photo attribution, and individual licenses are retained in the task records. Some photos have noncommercial, no-derivatives, or missing license information; public download access does not replace those terms.

## Direct image access

Use the exact `image_s3_uri` and `image_sha256` in each task. The HTTPS equivalent is:

```text
https://wildlife-csi-dataset-088543363904.s3.us-east-1.amazonaws.com/media/sha256/<first-two-hash-characters>/<sha256>.jpg
```

Individual image GETs and HEADs are public. Bucket contents and object versions cannot be listed anonymously. The runtime only fetches known keys and verifies their SHA-256; it never lists the bucket. Task metadata is distributed with this repository.

The bucket's reproducible settings are in [infra/s3-policy.json](../infra/s3-policy.json) and [infra/s3-public-access-block.json](../infra/s3-public-access-block.json). Maintainers with bucket-management credentials can apply them with:

```bash
aws s3api put-public-access-block \
  --bucket wildlife-csi-dataset-088543363904 \
  --public-access-block-configuration file://infra/s3-public-access-block.json
aws s3api put-bucket-policy \
  --bucket wildlife-csi-dataset-088543363904 \
  --policy file://infra/s3-policy.json
```

The policy grants only `s3:GetObject` under `media/sha256/`. Public ACLs remain blocked and ignored; there are no public listing or write grants.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

Tests use synthetic images and mocked model responses, so they need no provider credentials or paid model calls. The runtime is in [wildlife_csi](wildlife_csi/): CLI and run orchestration, provider adapters, S3 and local stores, and taxonomy scoring. Dataset-construction tools are not needed to run the frozen benchmark.
