# V4 Reuse Map — DJ Clustering v1

## Audit metadata

| Field | Value |
| :--- | :--- |
| Date | 2026-05-03 |
| Branch | feature/dj-clustering-v1 |
| Plan version | dj_clustering_plan.md (v6) |
| Task | D0.5 |

---

## V4 availability summary

| Status field | Value | Notes |
| :--- | :--- | :--- |
| v4_code_available | true | All source, tests, config, Slurm jobs, and docs present |
| v4_artifacts_available | false | Heavy artifacts (.npy, .parquet) are gitignored; not in repo |
| v4_playlist_status | v4_playlist_only | 5 playlist directories (V4_1–V4_5), 8 .m3u files present in Git |
| v4_competing | false | No reusable embedding or similarity space available in Git |
| v4_diagnostic_only | true | Reference code, playlists, and cluster labels only — not a fair leaderboard baseline |

---

## Legacy availability

| Directory | Present | Classification |
| :--- | :--- | :--- |
| legacy/v1/ | yes (scripts/, slurm/) | out_of_scope_for_v1 |
| legacy/v2/ | yes (docs/, models/, plans/, playlists/, results/, scripts/, slurm/, tests/) | out_of_scope_for_v1 |
| legacy/v3/ | yes (plans/, slurm/, src/) | out_of_scope_for_v1 |

Legacy V2 produced better subjective organization than V4 playlists (per plan hypothesis). This remains to be evaluated in D3–D5. No legacy artifacts are available in Git; only source and docs exist.

---

## V4 component classification table

| Path | Classification | Reason | Risk / Caveat | Import directly? | Revisit at |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/v4/common/audio_utils.py` | reuse_by_import | Audio loading, get_audio_files, DJ segmentation are format-agnostic | Uses soundfile.read (correct per LESSONS_LEARNED — torchaudio.load is broken on HPC a100 nodes) | Yes | D2.1 feature extraction |
| `src/v4/common/demucs_utils.py` | reuse_by_import | Stem separation (Demucs htdemucs) is pipeline-independent | Requires Apptainer SIF with demucs extras installed | Yes | D2.1 feature extraction |
| `src/v4/evaluation/metrics.py` | reuse_by_import | ARI, NMI, Recall@k, MRR, NDCG, pairwise_auc, transition_score, noise_rate are directly applicable to the DJ clustering sweep | Metric definitions must not be altered when reused | Yes | D5 sweep evaluation |
| `src/v4/common/embedding_utils.py` | adapt_into_src_dj_clustering | MERT-v1-330M embedder logic is relevant | Interface is V4-coupled; new pipeline needs different batching and sharding interface for D2 | Copy and adapt | D2.1 feature extraction |
| `src/v4/common/catalog.py` | adapt_into_src_dj_clustering | Track scanning and catalog-building logic is relevant | V4 catalog schema couples to V4 path_resolver; dj_clustering needs its own inventory schema with audio_content_hash, decode_status, canonical flags | Copy and adapt | D1.1–D1.3 inventory |
| `src/v4/common/logging_utils.py` | adapt_into_src_dj_clustering | JSONL logging and run manifest pattern is reusable | Minor V4-specific references to clean up when adapting | Copy and adapt | D1.1 or D2 |
| `src/v4/common/path_resolver.py` | diagnostic_reference_only | Shows HPC vs local path resolution pattern | V4-specific artifact layout; dj_clustering uses different artifact roots under artifacts/dj_clustering/ | Reference only | — |
| `src/v4/common/config_loader.py` | diagnostic_reference_only | Shows CLI > env > YAML cascade pattern | V4-coupled config key names | Reference only | D1.1 |
| `src/v4/pipeline/phase0_ingest.py` | diagnostic_reference_only | Reference for scan + validate + metadata merge pattern | V4-specific; D1 inventory rebuilds this more deterministically with hash-based identity | Reference only | D1 |
| `src/v4/pipeline/phase1_extract.py` | diagnostic_reference_only | Reference for Demucs + MERT + Essentia GPU extraction with sharding | V4-specific Slurm job structure | Reference only | D2 |
| `src/v4/pipeline/phase2_cluster.py` | diagnostic_reference_only | Reference for HDBSCAN + UMAP parameter choices | HDBSCAN/UMAP parameters are to be re-swept, not inherited | Reference only | D5 |
| `src/v4/pipeline/phase3_name.py` | diagnostic_reference_only | Reference for genre-voting cluster naming | Not needed until D6 export | Reference only | D6 |
| `src/v4/pipeline/phase4_order.py` | diagnostic_reference_only | Reference for transition-aware greedy NN ordering | Out of scope for v1 (ordered playlists not a target) | No | — |
| `src/v4/pipeline/phase5_export.py` | diagnostic_reference_only | Reference for M3U export and Windows path handling | V4-specific path logic; dj_clustering export rebuilt in D6 | Reference only | D6 |
| `src/v4/pipeline/phase1_merge_shards.py` | diagnostic_reference_only | Reference for shard consolidation pattern | V4-specific | Reference only | D2 |
| `src/v4/ui/app.py` | out_of_scope_for_v1 | Streamlit dashboard — not a v1 target | Regime 2 / UX task; re-evaluate only if D7 approves | No | Never (D0–D7) |
| `src/v4/adaptation/projection_head.py` | out_of_scope_for_v1 | Regime 2 projection head stub (MLP 1024→512→256) | Only if Regime 1 fails and evidence thresholds met per plan §7 D7 | No | D7 only if needed |
| `src/v4/adaptation/contrastive_trainer.py` | out_of_scope_for_v1 | Regime 2 contrastive trainer stub (NotImplementedError) | Same as above | No | D7 only if needed |
| `src/v4/evaluation/eval_runner.py` | diagnostic_reference_only | Reference for evaluation loop: load artifacts, compute metrics, save JSON | V4-specific artifact paths | Reference only | D5 |
| `slurm/jobs/v4/` (6 jobs) | diagnostic_reference_only | Reference Slurm job templates for GPU/CPU patterns | New jobs must derive from slurm/templates/generic_job.job per CLAUDE.md | Reference only | D2 Slurm jobs |
| `config/v4.yaml` | diagnostic_reference_only | Reference config structure | New pipeline gets configs/dj_clustering/ | Reference only | D1.1 |
| `playlists/V4_1–V4_5` (8 files) | diagnostic_reference_only | 5 exported playlist directories, 8 .m3u files. Paths are absolute (Windows format). Cannot map to current track_id without inventory. | Not ground truth per plan §3. Do not use for triplet boundary generation until D3.2 mapping is attempted. | No | D3.2 |
| `docs/v4/JOBS_STATUS.md` | diagnostic_reference_only | Reference for Slurm job status tracking | Not operational for dj_clustering | Reference only | — |
| `docs/v4/TODO.md` | diagnostic_reference_only | V4 progress reference | Not operational for dj_clustering | Reference only | — |
| `docs/V4_USAGE.md` | diagnostic_reference_only | Reference for end-to-end usage (Phase 0→5, UI, scaling) | Not operational for dj_clustering | Reference only | — |
| `tests/v4/` (6 files) | diagnostic_reference_only | Reference test patterns (integration and unit) | V4-specific fixtures; dj_clustering gets its own tests/ | Reference only | D1–D6 test tasks |
| `requirements_v4.txt` | diagnostic_reference_only | Reference dependency list | dj_clustering declares its own requirements in D2 | Reference only | D2 |
| `legacy/v1/`, `legacy/v2/`, `legacy/v3/` | out_of_scope_for_v1 | Older implementations | Legacy V2 clustering hypothesis deferred; evaluate in D5 only if reusable artifacts are loadable | No | D5 leaderboard only |

---

## Absent recommended files

`slurm/jobs/v4/phase0_ingest.job` is listed in the current `docs/PROJECT_MAP.md` but does not exist on disk. The ingest step appears to have been run inline or absorbed into other jobs. The 6 actual Slurm jobs present are: `eval_sweep.job`, `phase1_extract.job`, `phase1_extract_array.job`, `phase1_merge.job`, `phase2_to_5.job`, `smoke_test_gpu.job`. No action needed; record as absent.

---

## V4 playlist summary

- Directories present: V4_1, V4_2, V4_3, V4_4, V4_5 (5 directories)
- Total .m3u files: 8
- Path format: absolute, Windows-style (observed from PROJECT_MAP description)
- Private path contents: not printed and not committed
- Mapping status: cannot map to current track_id without a complete inventory; deferred to D3.2

---

## Frozen V4 policy (D0–D7)

V4 (`src/v4/`, `tests/v4/`, `config/v4.yaml`, `slurm/jobs/v4/`, `docs/v4/`, `docs/V4_USAGE.md`, `playlists/V4_*/`, `requirements_v4.txt`) is frozen for the duration of D0–D7.

Rules:
- V4 must not be modified, moved, deleted, renamed, or rewritten.
- New DJ clustering work must not be implemented under `src/v4/`.
- V4 is classified as `v4_diagnostic_only` for leaderboard purposes: no reusable embeddings or similarity space are available in Git.
- Selected V4 utilities (`audio_utils.py`, `demucs_utils.py`, `metrics.py`) are classified `reuse_by_import`; selected others (`catalog.py`, `embedding_utils.py`, `logging_utils.py`) are classified `adapt_into_src_dj_clustering`. Adaptation happens in D1–D2, not here.
- This frozen V4 policy and separate namespace are authorized by `docs/plans/dj_clustering_plan.md §6` and override the CLAUDE.md refactor-preference for D0–D7 only.

---

## Approved implementation namespace for DJ clustering

```
src/dj_clustering/
scripts/dj_clustering/
configs/dj_clustering/
tests/dj_clustering/
reports/dj_clustering/
```

These directories are created incrementally from D1 onward. None are created in D0.5.
