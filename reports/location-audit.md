# Country context audit

The country-aware suite contains all 2,000 vetted AnimalClue images. The five pinned Hugging Face `info_*.csv` files have taxonomy columns but no country or coordinates. The image filenames contain iNaturalist observation IDs, which were used to fetch public observation place IDs. Place records were resolved once and cached locally; the frozen `locations.jsonl` retains country, source, precision level, geoprivacy, and whether the country describes the observation itself.

| Provenance | Images | Country supplied to the common prompt |
| --- | ---: | --- |
| Public country place | 1,994 | Photo country is stated |
| Public country place; free-text locality disagrees | 1 | Mapped country Botswana is stated; Zimbabwe remains in metadata |
| Public Puerto Rico place | 1 | Country stated as United States; Puerto Rico retained as region |
| Private observation; no public place | 4 | Documented species-range country; provenance retained only in the dataset row |

The public country derivation includes eight observations whose iNaturalist country-level place is Taiwan (`admin_level: 0`, without a `place_type` value). The four private observations have neither public place IDs nor locality text. Their proxy country is United States, with source records for *Taxidea taxus* (two images), *Lontra canadensis*, and *Castor canadensis* in `species-range-examples.jsonl`. The model receives the same country prompt as every other task. These proxies are answer-derived hints and are flagged `is_observation_location: false` in the dataset; all four remain in the 2,000-task score denominator. Per-provenance results should be reported alongside the headline score.

The 169 obscured observations retain a country from public iNaturalist place records. [iNaturalist explains](https://help.inaturalist.org/en/support/solutions/articles/151000169938-what-is-geoprivacy-what-does-it-mean-for-an-observation-to-be-obscured-) that obscured observations share coarse geography, while private observations disclose no geographic information. The public records do not verify exact photo coordinates. This protocol uses only country names, not coordinates or finer locations.

The suite has 65 distinct country values and a frozen task hash recorded in `manifest.json`.
