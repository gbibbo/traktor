# Active Triplet Question Queue Summary (D4.2)

Aggregate-only report for the sweep-driven active triplet questions.
No track identifiers, question identifiers, paths, or library metadata
values are included.

## Selection method

Active questions are selected from disagreement among the top-3
executed Regime 1 first-sweep configs. Top-3 selection uses a
documented deterministic tie-break: sort executed (non-diagnostic)
leaderboard rows by triplet_accuracy descending, then by config_id
ascending, then take the first three. For a candidate triplet (A,B,C)
each config chooses B if d(A,B) < d(A,C); a disagreement exists when
at least one config chooses B and at least one chooses C. Candidates
are ranked by disagreement_margin = min(count_B, count_C) / 3, with
the variance of the B-vs-C rank difference as tie-break. Any shortfall
is filled with cluster-boundary candidates.

## Results

| Item | Value |
|---|---|
| Top configs compared | 3 |
| Candidate pool size | 1500 |
| Disagreement candidates found | 479 |
| Active questions requested | 85 |
| Active questions generated | 85 |
| From disagreement | 85 |
| From boundary fill | 0 |
| Removed: duplicates of existing queue | 5 |
| Removed: within-set / hash collisions | 628 |
| Existing queue questions | 40 |
| Total queue questions after update | 125 |
| Active sampling seed | 42 |
| Config-space rebuild seed | 13 |

## Status

Generated the full requested 85 active questions.

Active questions are candidates for future human comparison only.
No answers are produced in D4.2; answer ingestion is Task D4.3.
