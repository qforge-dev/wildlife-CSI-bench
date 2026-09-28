[![I Made AI Look at Poop. For Science. A robot examines animal droppings with a magnifying glass.](assets/og-poop.jpg)](https://labqoat.com/blog/what-animal-left-this)

# Wildlife CSI

**2,000 photos. Five kinds of animal trace. One question: what animal left this?**

Wildlife CSI tests whether vision models can identify an animal from what it leaves behind: droppings, footprints, feathers, eggs, and bones. Each model gets one photo and its country, then makes one species guess.

The best model in these seven runs identified **37.15%** of species correctly. **873 photos** went without a correct species guess from any model.

[Read the full post on the Labqoat blog](https://labqoat.com/blog/what-animal-left-this) · [Run it yourself](src/README.md) · [Chart data](assets/charts/results.json)

## Results

All seven models saw the same frozen suite with **high reasoning effort**. Scores include every photo, including failed calls and unusable answers.

| Model | Exact species | Genus or better | Family or better | Estimated run cost | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| **Claude Opus 5.5** | **37.15%** | **41.25%** | **51.75%** | $22.92 | [Run](runs/opus/) · [Scores](runs/opus/score/summary.json) |
| Muse Spark 1.3 Contributor | 33.35% | 37.55% | 48.80% | $0.77 | [Run](runs/muse/) · [Scores](runs/muse/score/summary.json) |
| GPT-6 Astra | 31.70% | 34.75% | 44.95% | $102.43 | [Run](runs/astra/) · [Scores](runs/astra/score/summary.json) |
| GLM 5.3 Flash | 21.75% | 25.20% | 35.80% | $1.73 | [Run](runs/glm/) · [Scores](runs/glm/score/summary.json) |
| GPT-6 Sol | 21.50% | 24.25% | 32.55% | $27.89 | [Run](runs/gpt-6-sol/) · [Scores](runs/gpt-6-sol/score/summary.json) |
| DeepSeek V4.1 Flash | 13.55% | 16.15% | 24.15% | $12.19 | [Run](runs/deepseek/) · [Scores](runs/deepseek/score/summary.json) |
| GPT-6 Luna | 8.20% | 10.05% | 15.45% | $1.78 | [Run](runs/luna/) · [Scores](runs/luna/score/summary.json) |

Each linked run contains all 2,000 prediction records, its configuration, per-photo scores, and scoring caches. Full provider records are compressed; [publication manifests](runs/index.json) document endpoint redaction and file checksums.

### Accuracy versus cost

Opus got **743 of 2,000** species right, 76 more than Muse. Muse's estimated run cost was **$0.77** at Contributor-tier rates; Astra cost the most while finishing behind both.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/cost-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/charts/cost-light.svg">
  <img src="assets/charts/cost-light.svg" alt="Exact-species accuracy versus estimated cost for seven models. Opus leads at 37.15% and $22.92; Muse reaches 33.35% at $0.77; Astra reaches 31.70% at $102.43. Full values are in the results table above." width="760">
</picture>

Cost uses a logarithmic scale. Estimates use recorded token usage and the rates configured for these runs, not invoices. They exclude calls without returned usage and separate scoring costs. The asterisk marks Muse's Contributor tier.

### Footprints were the hardest

Opus identified **205 of 400 egg photos (51.25%)**, but only **82 of 400 footprints (20.50%)**. It led on every trace type except feathers, where Muse scored highest at **38.75%**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/traces-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/charts/traces-light.svg">
  <img src="assets/charts/traces-light.svg" alt="Exact-species accuracy by trace type, with 400 photos per type. Opus leads on bones, eggs, droppings, and footprints; Muse leads on feathers. Footprints have the lowest accuracy for every model." width="760">
</picture>

Each cell shows exact-species accuracy on 400 photos. Darker cells mean more correct identifications; failures count as misses.

### Which clues stumped everyone?

Across all seven models, **873 of 2,000 photos (43.65%)** never received a correct species guess. That includes **265 footprints**, compared with **129 eggs**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/charts/coverage-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/charts/coverage-light.svg">
  <img src="assets/charts/coverage-light.svg" alt="Photos grouped by how many models identified the species. No model got 265 footprints, 164 bones, 161 droppings, 154 feathers, or 129 eggs right. Each trace type contains 400 photos." width="760">
</picture>

Each bar contains 400 photos. “Several” means more than one model but fewer than all seven.

[Download the exact counts and costs](assets/charts/results.json). GLM and DeepSeek scores are provisional because a text extractor contributed to some per-photo grades. DeepSeek's three assisted answers were all wrong, so its headline scores do not depend on those decisions.

## About the benchmark

Each model sees one photo and its country and is asked for one species name. Answers are checked against iNaturalist taxonomy, with cumulative credit for species, genus, and family. A text-only extractor handles replies the parser cannot resolve; it sees neither the photo nor the correct answer.

The images come from **AnimalClue** by [Risa Shinoda](https://huggingface.co/risashinoda) and collaborators: [droppings](https://huggingface.co/datasets/risashinoda/feces_yolo), [footprints](https://huggingface.co/datasets/risashinoda/footprint_yolo), [feathers](https://huggingface.co/datasets/risashinoda/feather_yolo), [eggs](https://huggingface.co/datasets/risashinoda/egg_yolo), and [bones](https://huggingface.co/datasets/risashinoda/bone_yolo). Frozen task records preserve source revisions, photo attribution, and licenses.

A species label does not guarantee that the species can be identified from a single trace photo, and there is no wildlife-expert baseline for this suite. Species are also unevenly represented: Opus's egg score falls from 51.25% to about 37% when each species gets equal weight. Four private observations use documented species-range countries instead of observed locations.

[Setup, model configuration, scoring, and dataset details](src/README.md)
