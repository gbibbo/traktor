# Metadata Quality Report

## Inventory basis

- Input artifact: `artifacts/dj_clustering/inventory/library_inventory.csv`
- Total rows: 246
- Canonical rows: 246
- Canonical decoded: 246
- Decode status: ok 246 (100.0%)
- Exact duplicate groups: 0

## Metadata quality flags

| Flag | Count |
|---|---|
| full_tags | 242 |
| partial_tags | 3 |
| no_tags | 1 |

## Metadata coverage (canonical decoded tracks)

| Field | Non-null | Coverage |
|---|---|---|
| artist | 245 | 99.6% |
| title | 245 | 99.6% |
| album | 242 | 98.4% |
| genre | 244 | 99.2% |
| label | 240 | 97.6% |
| year | 230 | 93.5% |
| bpm | 245 | 99.6% |
| key | 0 | 0.0% |

## Class balance and metadata sanity score validity

Applies the plan rule: a field is valid for `metadata_sanity_score` only if it has
≥ 2 classes and no single class covers > 85% of canonical decoded tracks.
Class names are not reported — only aggregate statistics.

| Field | Non-null | Classes | Dominant count | Dominant % | Valid for sanity | Reason |
|---|---|---|---|---|---|---|
| genre | 244 | 30 | 63 | 25.6% | yes | OK |
| artist | 245 | 209 | 10 | 4.1% | yes | OK |
| label | 240 | 153 | 37 | 15.0% | yes | OK |
| folder_hint | 246 | 2 | 243 | 98.8% | no | dominant class covers 98.8% > 85% threshold |

## BPM distribution (canonical decoded tracks)

- Non-null: 245 / 246 (99.6%)
- Valid as numeric feature (coverage ≥ 90%): yes

| Stat | Value |
|---|---|
| Min | 86.0 |
| Max | 140.0 |
| Mean | 123.2 |
| Std | 5.3 |

## Key coverage

- Non-null: 0 / 246 (0.0%)
- Assessment: key is absent from all tracks; excluded from feature candidates.

## Missingness profile (canonical decoded tracks)

| Condition | Count | % |
|---|---|---|
| Missing artist | 1 | 0.4% |
| Missing title | 1 | 0.4% |
| Missing artist AND title | 1 | 0.4% |
| Missing genre | 2 | 0.8% |
| Missing BPM | 1 | 0.4% |
| Missing key | 246 | 100.0% |

## Metadata sanity score validity summary

- Valid fields: genre, artist, label
- Invalid fields: folder_hint
- `metadata_sanity_score_unavailable`: false

## Suitability assessment

The inventory contains 246 canonical decoded tracks (small-N regime: 100 ≤ N < 500). Decode failure rate is 0.0% and exact duplicate count is 0. BPM coverage is 99.6%; BPM is available as a numeric feature for downstream grouping and sweep configurations. Key is absent from all tracks (0% coverage) and cannot be used as a feature or sanity signal in any downstream task. This is a data gap, not a blocker. 3 categorical field(s) pass the metadata sanity validity rule (≥2 classes, dominant class ≤85%) and may be used in metadata_sanity_score at D5. The dataset is sufficient to proceed to D1.4 and D2 feature extraction without manual metadata repair. Categorical metadata fields with class diversity may serve as auxiliary grouping signals but should not be treated as primary cluster labels, particularly given the absence of key data.

## Gate decision

| Criterion | Value | Result |
|---|---|---|
| N_canonical_decoded ≥ 100 | 246 | PASS |
| Decode failure rate < 20% | 0.0% | PASS |
| Exact duplicates | 0 | PASS |
| Key coverage | 0.0% | WARN (absent) |
| Sanity fields available | 3 valid | PASS |

**Overall: PASS+WARN**
- key coverage is 0% — key excluded from feature candidates
- folder_hint invalid for sanity scoring: dominant class covers 98.8% > 85% threshold

_Artifact basis: `artifacts/dj_clustering/inventory/library_inventory.csv` (git-ignored)_
