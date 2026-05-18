# DJ Clustering Progress

**Branch:** feature/dj-clustering-v1
**Integration branch:** main
**Current cut:** D3
**Current phase:** 3
**Current task:** D3.4
**Last completed:** D3.3
**Blocked:** No

---

## Cut D0 — Foundation

| Task | Description | Status |
|---|---|---|
| D0.1 | Repository inspection | DONE |
| D0.2 | Create and push feature/dj-clustering-v1 | DONE |
| D0.3 | Install plan files and create progress trackers | DONE |
| D0.4 | Configure ignored runtime artifact paths | DONE |
| D0.5 | Audit V4 reuse map and project namespace | DONE |

**D0 gate: PASSED** — branch, trackers, .gitignore, and V4 reuse map complete. Ready for Phase 1 / D1.1.

---

## Cut D1 — Inventory and identity foundation

| Task | Description | Status |
|---|---|---|
| D1.1 | Define local library inventory configuration | DONE |
| D1.2 | Implement library inventory script | DONE |
| D1.3 | Create metadata quality report | DONE |

### D1.3 results

| Metric | Value |
|---|---|
| Report | `reports/dj_clustering/metadata_quality_report.md` |
| Gate | PASS+WARN |
| N_canonical_decoded | 246 (small-N regime) |
| Decode failure rate | 0.0% |
| Exact duplicates | 0 |
| Key coverage | 0% (absent — excluded from features) |
| Valid sanity fields | genre, artist, label (3 fields) |
| Invalid sanity field | folder_hint (98.8% dominant class) |
| metadata_sanity_score_unavailable | false |
| BPM coverage | 99.6% (valid as numeric feature) |

Warnings: key absent (data gap, not a blocker); folder_hint invalid for sanity scoring (expected — single library root).
D1 gate PASSED.

---

## Cut D2 — Feature extraction foundation

| Task | Description | Status |
|---|---|---|
| D2.1 | Create feature extraction configuration | DONE |
| D2.2 | Implement feature extraction | DONE |
| D2.3 | Evaluate V2 and current V4 availability | DONE |

**D2 gate: PASSED** — default features, feature_manifest, feature_quality_report, and reference_availability_report complete. Ready for Cut D3 / D3.1.

---

## Cut D3 — Pair features and evaluation evidence

| Task | Description | Status |
|---|---|---|
| D3.1 | Build pairwise feature table | DONE |
| D3.2 | Generate initial manual triplet question queue | DONE |
| D3.3 | Ingest manual triplet answers | DONE |

### D3.3 results (manual triplet answer ingest)

| Item | Value |
|---|---|
| Library module | `src/dj_clustering/triplet_ingest.py` |
| Driver script | `scripts/dj_clustering/ingest_manual_triplets.py` |
| Manual triplets artifact | `artifacts/dj_clustering/triplets/manual_triplets.csv` (ignored) |
| Skip log artifact | `artifacts/dj_clustering/triplets/triplet_skip_log.csv` (ignored) |
| Summary report | `reports/dj_clustering/manual_triplets_summary.md` (committed) |
| Total questions | 40 |
| Non-skip answers | 37 |
| Skip answers | 3 |
| Answer distribution | B = 12, C = 25, skip = 3 |
| Decision rule (non-skip ≥ 20) | PASS — evidence available for sweep |
| Answer normalization | Case-insensitive (b/B→B, c/C→C, skip/Skip/SKIP→skip); other values rejected |
| Tests passed | 25/25 (test_triplet_ingest.py), 163/163 full dj_clustering suite |
| Artifacts ignored | PASS (.gitignore: `artifacts/`) |
| Summary privacy guard | PASS (aggregate-only; no IDs, notes, or metadata tokens) |
| Tracker blocked | false |

**Next:** D3.4 — Create 1001Tracklists matching input.

---

### D3.2 results

| Item | Value |
|---|---|
| Triplet queue CSV | `artifacts/dj_clustering/triplets/triplet_question_queue.csv` (ignored) |
| Triplet queue M3U | `artifacts/dj_clustering/triplets/triplet_question_queue.m3u` (ignored) |
| Summary report | `reports/dj_clustering/triplet_queue_summary.md` |
| V4 mapping report | `reports/dj_clustering/v4_playlist_mapping_report.md` |
| Total questions | 40 |
| NN questions (mert_full_knn) | 20 |
| Boundary questions (v4_boundary) | 20 |
| V4 coverage | 100.0% (239/239 entries mapped, gate PASS) |
| Boundary source | v4_boundary (V4_5, 8 L1 groups) |
| Seed | 42 |
| KNN k | 50 |
| Default embedding | mert_full / last_layer_mean |
| De-duplication removed | 0 |
| Tests | 27/27 passed (test_triplets.py) |
| Privacy guard | PASS (both committed reports) |

D3.2 sub-steps:
- module: `src/dj_clustering/triplets.py`
- driver: `scripts/dj_clustering/generate_triplet_questions.py`
- tests: `tests/dj_clustering/test_triplets.py`

### D3.1 results

| Item | Value |
|---|---|
| Pair features artifact | `artifacts/dj_clustering/pairs/pair_features.parquet` (ignored) |
| Pair manifest artifact | `artifacts/dj_clustering/pairs/pair_manifest.json` (ignored) |
| Summary report | `reports/dj_clustering/pair_features_summary.md` |
| N_canonical_decoded | 246 |
| Expected pairs | 30135 |
| Actual pairs | 30135 |
| Pair_id format | `{track_id_a}__{track_id_b}` with `a < b` |
| Default embedding source | mert_full + last_layer_mean |
| Cosine families stored | 9 (3 sources × 3 aggregations) |
| MERT pair availability (each combo) | 29890 / 30135 |
| BPM pair availability | 29890 / 30135 |
| BPM source | inventory `bpm` column (raw BPM units) |
| BPM tolerance | 8.0 BPM |
| Metadata pair availability | 29890 / 30135 |
| Metadata sanity fields | 3 (per `pairs.metadata_sanity_fields`) |
| Committed profiles | audio_only, audio_plus_metadata_light |
| Omitted profiles | audio_plus_bpm_key_light, audio_plus_bpm_key_metadata_light (key_compat unavailable) |
| Tests | 25/25 passed |
| Range checks | all similarity columns ∈ [0, 1] when available |

D3.1 sub-steps:
- modules: `src/dj_clustering/pair_features.py`, `src/dj_clustering/similarity_profiles.py`
- driver: `scripts/dj_clustering/build_pair_features.py`
- tests: `tests/dj_clustering/test_pair_features.py`, `tests/dj_clustering/test_similarity_profiles.py`
- config update: `configs/dj_clustering/features.yaml` (`pairs:` block added)
- privacy guard: PASS on `reports/dj_clustering/pair_features_summary.md`
- raw BPM is the source for tempo equivalence; normalized BPM is used only as an aligned cross-check (`bpm_norm_abs_diff` column).

### D2.3 results

| Item | Value |
|---|---|
| Report | `reports/dj_clustering/reference_availability_report.md` |
| V2 status | diagnostic_only (cluster IDs + UMAP coords + playlists; no embeddings) |
| V2 embeddings | unavailable (rerun forbidden by plan §3) |
| V2 playlists | 17 (counts only) |
| V4 code available | true |
| V4 artifacts available | false |
| V4 playlist status | v4_playlist_only (5 directories, 72 m3u files) |
| V4 competing baseline | false |
| V4 diagnostic only | true |
| Cut D2 gate | PASSED |
| Next plan-defined task | D3.1 (build pairwise feature table) |

### D2.2 results

| Metric | Value |
|---|---|
| Report | `reports/dj_clustering/feature_quality_report.md` |
| Gate | PASS (mert_full failure rate 1.2% ≤ 10%) |
| Tracks processed | 246 |
| Segments processed | 738 (3 per track, all `long` class) |
| MERT backend | validated (Apptainer pytorch_2.7.0_cu128 on a100) |
| HPSS available | true |
| mert_full | 245 tracks × 3 aggregations × 1024 dim, L2-normalized |
| mert_perc | 245 tracks × 3 aggregations × 1024 dim, L2-normalized |
| mert_concat | 245 tracks × 3 aggregations × 2048 dim, L2-normalized |
| BPM | 246 tracks × 1 dim (1 imputed) |
| mert_full failure rate | 1.2% (3 failed manifest rows / 246 tracks) |
| Aggregations | last_layer_mean, last_4_layers_mean, layer_7_mean |

D2.2 sub-phases:
- D2.2a (a58a9fd): scaffold + segment_manifest (738 rows)
- D2.2b (25b156f, ef1a21e): MERT backend validation in Apptainer
- D2.2c (bc2a4ce): smoke extraction (3 tracks) — all sources passed
- D2.2d (job 2126808): full extraction on 246 tracks — gate PASS

### D2.1 results

| Item | Value |
|---|---|
| Config | `configs/dj_clustering/features.yaml` |
| MERT model | m-a-p/MERT-v1-330M |
| MERT backend | planned_hpc_apptainer (D2.2 must validate) |
| essentia | not installed → continue_without_essentia |
| demucs | not installed → HPSS default |
| HPSS backend | requires D2.2 validation (scipy_numpy preferred) |
| Percussion fallback | omit_mert_perc_and_mert_concat if HPSS unavailable |
| BPM feature | enabled (99.6% coverage) |
| key feature | disabled (0% coverage) |
| genre/artist/label | sanity evaluation only, not embedding input |

### D1.2 results

| Metric | Value |
|---|---|
| Total files scanned | 492 |
| Audio files (whitelist) | 246 |
| Non-audio excluded | 246 |
| Decode ok | 246 (100%) |
| Decode failed | 0 (0.0%) |
| Unique audio hashes | 246 |
| Exact duplicates | 0 |
| Canonical decoded (N_canonical_decoded) | 246 |
| Metadata full_tags | 242 |
| Metadata partial_tags | 3 |
| Metadata no_tags | 1 |

Backends used: soundfile (primary), mutagen 1.47.0 (ID3 metadata).
Tests: 37/37 passed.
Decode guard: PASS (0.0% < 20% threshold).

---

## Regime 1 snapshot

| Metric | Value |
|---|---|
| manual_triplets_answered | 0 |
| validated_1001_positives | 0 |

---

## Reference status (resolved in D0.5, finalized in D2.3)

| System | Status |
|---|---|
| V2 | diagnostic_only |
| V2 embeddings | unavailable |
| V2 result tables | available (cluster IDs + UMAP coords) |
| V2 playlists | 17 m3u files (counts only) |
| V4 | diagnostic_reference_only |
| v4_code_available | true |
| v4_artifacts_available | false |
| v4_playlist_status | v4_playlist_only (5 dirs, 72 m3u files) |
| v4_competing | false |
| v4_diagnostic_only | true |

_Last updated: D3.3 (Cut D3 active; advancing to D3.4)._
