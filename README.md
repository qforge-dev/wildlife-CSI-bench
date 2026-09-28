[![I Made AI Look at Poop. For Science.](assets/og-poop.jpg)](https://labqoat.com/blog/what-animal-left-this)

# Wildlife CSI

**2,000 photos · 5 trace types · 7 vision models**

Which animal left this footprint, dropping, feather, egg, or bone? Each model gets one photo and its country, then guesses the species.

[Blog post](https://labqoat.com/blog/what-animal-left-this) · [Run it yourself](src/README.md) · [All runs](runs/) · [Chart data](assets/charts/results.json)

## Accuracy versus cost

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/cost-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/charts/cost-light.svg">
  <img src="assets/charts/cost-light.svg" alt="Exact-species accuracy versus estimated cost for seven models. Opus leads at 37.15% and $22.92; Muse reaches 33.35% at $0.77; Astra reaches 31.70% at $102.43. Full values are in the expandable results table below." width="760">
</picture>

Exact-species accuracy across all 2,000 photos, including failed calls. Costs are estimates; Muse uses Contributor pricing.

<details>
<summary>Scores and run files</summary>

| Model | Exact species | Genus or better | Family or better | Estimated run cost | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| **Claude Opus 5.5** | **37.15%** | **41.25%** | **51.75%** | $22.92 | [Run](runs/opus/) · [Scores](runs/opus/score/summary.json) |
| Muse Spark 1.3 Contributor | 33.35% | 37.55% | 48.80% | $0.77 | [Run](runs/muse/) · [Scores](runs/muse/score/summary.json) |
| GPT-6 Astra | 31.70% | 34.75% | 44.95% | $102.43 | [Run](runs/astra/) · [Scores](runs/astra/score/summary.json) |
| GLM 5.3 Flash | 21.75% | 25.20% | 35.80% | $1.73 | [Run](runs/glm/) · [Scores](runs/glm/score/summary.json) |
| GPT-6 Sol | 21.50% | 24.25% | 32.55% | $27.89 | [Run](runs/gpt-6-sol/) · [Scores](runs/gpt-6-sol/score/summary.json) |
| DeepSeek V4.1 Flash | 13.55% | 16.15% | 24.15% | $12.19 | [Run](runs/deepseek/) · [Scores](runs/deepseek/score/summary.json) |
| GPT-6 Luna | 8.20% | 10.05% | 15.45% | $1.78 | [Run](runs/luna/) · [Scores](runs/luna/score/summary.json) |

GLM and DeepSeek scores are provisional; see their run summaries for extractor-assisted grades.

</details>

## By trace type

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/traces-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/charts/traces-light.svg">
  <img src="assets/charts/traces-light.svg" alt="Exact-species accuracy by trace type, with 400 photos per type. Opus leads on bones, eggs, droppings, and footprints; Muse leads on feathers. Footprints have the lowest accuracy for every model." width="760">
</picture>

400 photos per type. Footprints were hardest for every model.

## Which clues stumped everyone?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/coverage-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/charts/coverage-light.svg">
  <img src="assets/charts/coverage-light.svg" alt="Photos grouped by how many models identified the species. No model got 265 footprints, 164 bones, 161 droppings, 154 feathers, or 129 eggs right. Each trace type contains 400 photos." width="760">
</picture>

**873 photos** received no correct species guess from any of the seven models.

Photos from [AnimalClue](https://huggingface.co/risashinoda). Attribution and licenses are preserved in the task records.
