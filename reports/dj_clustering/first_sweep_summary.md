# Regime 1 First Sweep Summary (D4.1)

Aggregate-only report for the capped Regime 1 exploratory similarity sweep.
No track identifiers, comparison identifiers, row-level answers, or library
metadata values are included.

## Run scope

| Item | Value |
|---|---|
| Task | D4.1 — Regime 1 sweep runner, capped first sweep |
| Sweep grid | Vector grid (frozen MERT embeddings); no similarity-profile axis |
| Total non-baseline grid configs | 2304 |
| Seed-13 sampling cap | 200 |
| Sampled grid configs run | 200 |
| Fixed baselines run | 3 |
| Executable configs run | 203 |
| Diagnostic-only reference rows | 2 (not executed) |
| Total leaderboard rows | 205 |
| Failed configs | 0 |
| Wall time | ~40 s (CPU) |

Leaderboard convention: diagnostic-only reference rows are appended to the
leaderboard (205 rows = 203 executed + 2 diagnostic).

## Evidence basis

| Item | Value |
|---|---|
| Manual triplet non-skip answers | 37 |
| Manual triplet skip rows | 3 |
| Triplets scored per executed config | 37 |
| Evidence classification | exploration |
| 1001Tracklists status | no usable source — excluded from metrics, non-blocking |

Triplet scoring: a config predicts the closer candidate from Euclidean distance
in its own transformed embedding space; answer `B`/`C` direction as defined in
the operational plan; skip rows excluded.

## Triplet accuracy (executed configs)

| Statistic | Value |
|---|---|
| Minimum | 0.405 |
| Median | 0.541 |
| Maximum | 0.622 |
| Fixed baselines | 0.568 / 0.595 / 0.622 (perc / full / concat HDBSCAN defaults) |

The strongest accuracy in this exploratory sweep comes from a fixed HDBSCAN
baseline (0.622).

## Cluster diagnostics (executed configs)

| Diagnostic | Range |
|---|---|
| Cluster count | 0 – 22 |
| Largest cluster share | 0.00 – 0.93 |
| Singleton count | 0 – 10 |
| HDBSCAN raw noise rate | 0.00 – 1.00 |

Executed configs by clusterer: HDBSCAN 137, KMeans 34, agglomerative 32. The
wide diagnostic ranges (degenerate all-noise and giant-cluster configs both
appear) are expected for an unfiltered exploratory grid.

## Winner selection

No winner is selected in D4.1. This is an exploratory diagnostic sweep; winner
selection belongs to the later plan-defined full-sweep / winner-selection
tasks. The 37 non-skip triplets satisfy the minimum exploration threshold but
results here are not used to declare a Regime 1 winner.

## Output artifacts (ignored runtime paths)

- `runs/dj_clustering/first_sweep/first_sweep_leaderboard.csv`
- `runs/dj_clustering/first_sweep/component_metrics.csv`
- `runs/dj_clustering/first_sweep/run_metadata.json`
- `runs/dj_clustering/first_sweep/resolved_sweep_config.yaml`

These are runtime artifacts under the ignored `runs/` tree and are not committed.
