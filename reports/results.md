# Wildlife CSI results

All runs below use the frozen 2,000-image country-aware suite, task hash `b1d0ee38219c2f63542ef6d39249ab2735b976287bb775101f5eba73cd877b42`, and high reasoning effort. They have 2,000 unique prediction records and complete scores with the same scoring logic. Percentages use all 2,000 tasks as the denominator; genus and family columns include higher-rank successes.

| Model | Species exact | Genus or better | Family or better | Unanswered or invalid | Generation time | Estimated model cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Muse Spark 1.3 Contributor | 33.35% | 37.55% | 48.80% | 0 | 1h 06m | $0.77 |
| GLM 5.3 Flash | 21.75% | 25.20% | 35.80% | 4 invalid answers | 19m 50s | $1.73 |
| GPT-6 Sol | 21.50% | 24.25% | 32.55% | 99 provider rejections | 42m 39s | $27.89 |
| DeepSeek V4.1 Flash | 13.55% | 16.15% | 24.15% | 5 empty, 61 truncated | 2h 17m | $12.19 |
| GPT-6 Luna | 8.20% | 10.05% | 15.45% | 102 provider rejections, 2 invalid answers | 41m 29s | $1.78 |

The provider rejections in the Azure runs were content-policy responses. DeepSeek's truncated calls exhausted its configured 16,000-token output limit. All failures remain in the denominator. Model costs use rates in the YAML configs and are estimates, not invoices. OpenRouter reported $0.77 for Muse, $0.75 for GLM, and $9.15 for DeepSeek in the saved usage records; Azure did not return billed costs.

| Model | Input tokens | Output tokens | Review items | Extractor-assisted |
| --- | ---: | ---: | ---: | ---: |
| Muse | 2,952,668 | 2,354,492 | 2 | 0 |
| GLM | 3,029,410 | 268,482 | 12 | 2 |
| GPT-6 Sol | 2,555,699 | 2,277,629 | 1 | 0 |
| DeepSeek | 1,462,393 | 9,794,661 | 4 | 3 |
| GPT-6 Luna | 1,427,477 | 3,277,969 | 5 | 0 |

The scorer first parses the requested `ANIMAL:` answer, then resolves names against iNaturalist taxonomy IDs. A text-only extractor quotes a name only when the parser cannot establish a match; its calls are retained in `score/extractions.jsonl`. The summary marks GLM and DeepSeek provisional because an extracted name contributed to their per-task grades. DeepSeek's three extractor-assisted answers were all graded wrong, so its headline scores do not depend on those decisions. Inspect each model's `score/review_queue.jsonl` before publication.

The task records mark four answer-derived country examples for private observations. The score summary reports them separately by location basis. The [location audit](location-audit.md) explains their provenance and the other country exceptions. Photo rights and whether a trace is visually identifiable still need a publication review.

For each model, `data/benchmarks/wildlife-csi-country-v1-2000/runs/<model>/` holds the raw predictions, run manifest, per-task scores, resolution and extraction records, and summary. The summaries include task, prediction, resolution, extraction, and code hashes. The private S3 bucket contains image bytes only.
