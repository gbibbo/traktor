# Feature Quality Report

| Metric | Value |
|---|---|
| N_canonical_decoded | 246 |
| N_segments_generated | 738 |
| HPSS available | True |
| Smoke run | False |

## Per-source extraction summary

| Source | Aggregation | N_extracted | N_failed | N_omitted | Failure rate | Dim |
|---|---|---|---|---|---|---|
| mert_full | last_layer_mean | 245 | 1 | 0 | 0.4% | 1024 |
| mert_full | last_4_layers_mean | 245 | 1 | 0 | 0.4% | 1024 |
| mert_full | layer_7_mean | 245 | 1 | 0 | 0.4% | 1024 |
| mert_perc | last_layer_mean | 245 | 0 | 1 | 0.0% | 1024 |
| mert_perc | last_4_layers_mean | 245 | 0 | 1 | 0.0% | 1024 |
| mert_perc | layer_7_mean | 245 | 0 | 1 | 0.0% | 1024 |
| mert_concat | last_layer_mean | 245 | 0 | 1 | 0.0% | 2048 |
| mert_concat | last_4_layers_mean | 245 | 0 | 1 | 0.0% | 2048 |
| mert_concat | layer_7_mean | 245 | 0 | 1 | 0.0% | 2048 |

## Metadata features

| Feature | N_valid | N_null_imputed |
|---|---|---|
| bpm | 245 | 1 |

## Gate

| Check | Result |
|---|---|
| mert_full failure rate ≤ 10% | PASS (1.2%) |
