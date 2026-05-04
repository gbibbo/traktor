# Pair Feature Summary (D3.1)

Aggregate-only report. The pair feature table itself
(`pair_features.parquet`) and its accompanying manifest are runtime
artifacts and are not committed to Git.

## Pair universe

- Canonical decoded tracks: **246**
- Expected unordered pairs (N · (N−1) / 2): **30135**
- Computed pairs: **30135**
- Pair count matches the policy: **YES**

The pair universe contains every canonical decoded track. Tracks lacking a
specific feature stay in the universe; only the dependent feature columns
flag them with `*_available = False`. No pairs are dropped.

## Pair construction

- Track identifiers are sorted ascending (lexicographic) before enumeration.
- Every unordered pair `(a, b)` is emitted exactly once with `a < b`.
- `pair_id = "{track_id_a}__{track_id_b}"` (double-underscore separator).
- `pair_index` runs `0 .. 30134` in row-major upper-triangle order.

Track identifiers are hashed IDs from D1.1; they are not enumerated in this
report.

## Cosine pair feature coverage

For each MERT source × aggregation pair, `available = True` iff both
endpoints have a stored embedding. Pairs marked unavailable carry `NaN` in
`cosine_similarity`, `cosine_distance`, and `cosine_embedding_01`.

| Source       | Aggregation          | Available pairs | Missing pairs | Coverage |
|--------------|----------------------|-----------------|---------------|----------|
| mert_full    | last_layer_mean      | 29890           | 245           | 99.19%   |
| mert_full    | last_4_layers_mean   | 29890           | 245           | 99.19%   |
| mert_full    | layer_7_mean         | 29890           | 245           | 99.19%   |
| mert_perc    | last_layer_mean      | 29890           | 245           | 99.19%   |
| mert_perc    | last_4_layers_mean   | 29890           | 245           | 99.19%   |
| mert_perc    | layer_7_mean         | 29890           | 245           | 99.19%   |
| mert_concat  | last_layer_mean      | 29890           | 245           | 99.19%   |
| mert_concat  | last_4_layers_mean   | 29890           | 245           | 99.19%   |
| mert_concat  | layer_7_mean         | 29890           | 245           | 99.19%   |

Missing pairs in every source/aggregation: 245 = C(N, 2) − C(N − 1, 2). One
track failed feature extraction in D2.2; all 245 pairs that include it carry
`NaN` MERT cosines.

## BPM pair feature coverage (raw BPM)

- Source: `bpm` column of the canonical inventory CSV (units: real BPM).
- Tolerance: `bpm_tolerance_bpm = 8.0` BPM (committed in `configs/dj_clustering/features.yaml`).
- Tempo equivalence: `bpm_eff_abs_diff = min(|a − b|, |a − 2b|, |a − 0.5b|, |2a − b|, |0.5a − b|)`.
- Similarity: `bpm_similarity = 1 − clip(bpm_eff_abs_diff / 8.0, 0, 1)`.
- Available pairs: **29890** (245 missing because one canonical track has no BPM in the inventory).

`bpm_eff_abs_diff` distribution (available pairs only, BPM):

| Stat | Value |
|------|-------|
| min  | 0.00  |
| p25  | 1.00  |
| p50  | 3.00  |
| p75  | 6.00  |
| max  | 35.00 |
| mean | 4.72  |

`bpm_similarity` distribution (available pairs only):

| Stat | Value |
|------|-------|
| min  | 0.000 |
| p25  | 0.250 |
| p50  | 0.625 |
| p75  | 0.875 |
| max  | 1.000 |
| mean | 0.542 |

The D2.2 normalized BPM artifact is not used for tempo equivalence. An
optional `bpm_norm_abs_diff` column is attached as an aligned cross-check
only and never feeds the similarity profiles.

## Metadata pair feature coverage

- Sanity-validated metadata fields (committed in `configs/dj_clustering/features.yaml: pairs.metadata_sanity_fields`): three fields, ordered as in the config. Field-level statistics in this report use the indices `field_1`, `field_2`, `field_3` matching the same config order. The validation rule and field set are defined in [reports/dj_clustering/metadata_quality_report.md](metadata_quality_report.md).
- Per-pair similarity averages binary matches over the fields available on both sides; when no field is available on both sides, similarity is 0 and `metadata_similarity_available = False`.

| Field   | Available pairs |
|---------|-----------------|
| field_1 | 29646           |
| field_2 | 29890           |
| field_3 | 28680           |

`metadata_similarity_available = True` count: **29890**.

`metadata_similarity` distribution (available pairs only):

| Stat | Value |
|------|-------|
| min  | 0.000 |
| p25  | 0.000 |
| p50  | 0.000 |
| p75  | 0.000 |
| max  | 1.000 |
| mean | 0.047 |

The values in this column reflect identity-style matching across sanity
fields, so most pairs of distinct items score 0; the long tail toward 1
captures pairs that share the same value on most available fields.

## Default embedding source

- Default embedding source: **mert_full**
- Default aggregation: **last_layer_mean**

Committed in `configs/dj_clustering/features.yaml: pairs.default_embedding_source` and `pairs.default_embedding_aggregation`. This default matches the `default_aggregation` in the same config and the `MERT_full_HDBSCAN_default` baseline referenced in Section 14 of the operational plan. The committed similarity profiles reference this default.

## Profile inventory

| Profile                                | Status     | Reason                  |
|----------------------------------------|------------|-------------------------|
| audio_only                             | committed  | —                       |
| audio_plus_bpm_key_light               | omitted    | key_compat unavailable  |
| audio_plus_metadata_light              | committed  | —                       |
| audio_plus_bpm_key_metadata_light      | omitted    | key_compat unavailable  |

Key coverage in the dataset is 0.0% (D1.3), so `key_compat` cannot be
computed; the two key-dependent profiles are omitted per the plan rule "if a
feature source is missing, omit only the dependent profile and record it".

`profile_audio_only_similarity` distribution (available pairs only):

| Stat | Value |
|------|-------|
| min  | 0.770 |
| p25  | 0.915 |
| p50  | 0.931 |
| p75  | 0.944 |
| max  | 0.992 |
| mean | 0.928 |

`profile_audio_plus_metadata_light_similarity` distribution (available pairs only):

| Stat | Value |
|------|-------|
| min  | 0.654 |
| p25  | 0.779 |
| p50  | 0.794 |
| p75  | 0.809 |
| max  | 0.993 |
| mean | 0.795 |

## Range checks

All cosine, BPM, metadata, and committed-profile similarity columns are
inside `[0, 1]` on every pair where the corresponding `available` flag is
`True`. `*_distance` columns equal `1 − *_similarity` exactly, by
construction. The build script asserts these invariants before writing.

## Cross-references

- [reports/dj_clustering/feature_quality_report.md](feature_quality_report.md) — D2.2 feature artifact statuses (one MERT failure across 1.2% of canonical decoded tracks).
- [reports/dj_clustering/metadata_quality_report.md](metadata_quality_report.md) — D1.3 metadata sanity validation that selects the three fields used in the metadata-similarity computation.
- [reports/dj_clustering/reference_availability_report.md](reference_availability_report.md) — Reference status for V2 / V4 (informational).
- [docs/plans/dj_clustering_plan.md](../../docs/plans/dj_clustering_plan.md) — Section 12 "Task D3.1. Build pairwise feature table".

## Provenance

- Operational plan: `docs/plans/dj_clustering_plan.md` (D3.1).
- Feature config: `configs/dj_clustering/features.yaml` (`pairs:` block).
- Inventory config: `configs/dj_clustering/inventory.yaml`.
- Source modules: `src/dj_clustering/pair_features.py`, `src/dj_clustering/similarity_profiles.py`.
- Driver script: `scripts/dj_clustering/build_pair_features.py`.
- Tests: `tests/dj_clustering/test_pair_features.py`, `tests/dj_clustering/test_similarity_profiles.py`.
- Pair table and manifest are runtime artifacts under
  `artifacts/dj_clustering/pairs/` and are excluded from Git.
