# DJ Clustering — Cluster-Exit & Portability Checkpoint

**Status:** complete
**Exit block:** EXIT.1–EXIT.7
**Surrey access ends:** 2026-05-31
**Branch:** feature/dj-clustering-v1

## 1. Why this checkpoint exists

Surrey HPC access ends on 2026-05-31. This checkpoint freezes a portable,
checksummed copy of every artifact needed to resume the DJ clustering project
off the cluster — on a local CPU host, a paid GPU VM, or another HPC — without
re-running any expensive cluster step.

This is an operational migration block. It makes **no scientific claim** and
advances **no science task**. The scientific operational plan
(`docs/plans/dj_clustering_plan.md`, cuts D0–D7) is unchanged. Task D4.3
remains blocked: collecting the active triplet answers is an offline human
task, not a cluster dependency.

## 2. Exit block

| ID | Action |
|---|---|
| EXIT.1 | Freeze repo state and verify the branch is clean |
| EXIT.2 | Inventory the critical ignored artifacts |
| EXIT.3 | Generate sha256 checksums and the restore manifest |
| EXIT.4 | Package the critical artifacts into portable archives |
| EXIT.5 | Document local / paid-cloud / HPC restore paths |
| EXIT.6 | Amend the tracker with a migration section; keep D4.3 blocked |
| EXIT.7 | Commit, push, then create the annotated exit tag |

The EXIT.* items are migration tasks. They are deliberately **not** appended
to the science task history in the progress tracker.

## 3. What is preserved in GitHub

Code (`src/dj_clustering/`, `scripts/dj_clustering/`), tests
(`tests/dj_clustering/`), configs (`configs/dj_clustering/`), all small
reports (`reports/dj_clustering/`), the operational plan, the progress
trackers, `docs/PROJECT_MAP.md`, and these migration documents — including the
archive checksums and the verification-safe raw-audio aggregate. Together these
let anyone verify a restored payload byte-for-byte even though the payload
itself is not in Git.

## 4. What is NOT committed

Audio; triplet question/answer CSV and M3U review files; `.npy` embeddings;
the pairwise Parquet table; run outputs (CSV/JSON/YAML); container images;
model caches; logs; and any file carrying private collection metadata. All are
already covered by `.gitignore` (`artifacts/`, `runs/`, `data/`, `.cache/`,
audio and binary suffixes). The portable archives and the per-file raw-audio
checksum manifest are written under the ignored export directory and are never
committed.

## 5. Critical resume payload

Bundled into the critical archive (repo-relative member directories):

- `artifacts/dj_clustering/inventory/` — track inventory (identity, decode
  status, metadata, BPM, dedup state)
- `artifacts/dj_clustering/features/` — frozen MERT embeddings (full, perc,
  concat × three aggregations), BPM metadata vector, segment and feature
  manifests
- `artifacts/dj_clustering/pairs/` — pairwise feature table and its manifest
- `artifacts/dj_clustering/triplets/` — triplet question queue and review
  M3U, the human comparison answers, the skip log, and the answer templates
  (including the active answer template still awaiting human input)
- `artifacts/dj_clustering/tracklists_1001/` — the metadata-only input template
- `runs/dj_clustering/first_sweep/` — the exploratory first-sweep results
  (ranking table, component metrics, run metadata, resolved config)

The first-sweep results are **exploratory** evidence only. No winner is
selected and no final claim is made until D4.3/D5 evidence is restored and run.

## 6. Optional payload

Bundled into the optional archive: `artifacts/dj_clustering/features_smoke/`
and `runs/dj_clustering/first_sweep_validation/`. These reproduce convenience
and validation-subset outputs and can be dropped on constrained targets.

## 7. Portable archives (this checkpoint)

| Archive | Bytes | sha256 |
|---|---|---|
| `dj_clustering_critical_20260528.tar.zst` | 14987165 | `071e5e919a63a7bc51277834f4efbbf19840fc30fb75e211fca7eb40329489ea` |
| `dj_clustering_optional_20260528.tar.zst` | 140445 | `02e56662898233e530a02c36747ecc39a4b781db05751b3ecb514e22bd6fc0be` |

Both live under `artifacts/dj_clustering/_export/` (ignored). The exporter that
produces them is `scripts/dj_clustering/export_portable_state.py`, and the
machine-readable member list is `dj_clustering_artifact_manifest.yaml`.

## 8. Raw audio handling

The source collection (492 files, 4,415,100,271 bytes) is **never committed**
and **never moved** by this checkpoint. A per-file sha256 manifest is generated
under the ignored export directory
(`artifacts/dj_clustering/_export/raw_audio_checksums_20260528.txt`,
sha256 `daed5d8e422b77420ec5859f45c959923236ea45ad42e7e94ec7cc09e40d0ee8`).
That manifest carries collection paths and therefore stays ignored. Only the
aggregate (count, total bytes, manifest sha256) is recorded in committed docs.

The audio bytes travel out-of-band to a user-chosen destination
(`user_decision_out_of_band`); see the restore guide. The committed manifest
sha256 plus the inventory content hashes are enough to re-link restored audio
and detect drift.

## 9. Cluster-independent continuation

- **CPU-only, no GPU:** D4.3 answer ingestion, all report updates, and the
  clustering sweeps — because the embeddings and pairwise table already exist
  as artifacts. This is the primary post-exit path.
- **GPU required only** for MERT re-extraction or any Regime-2 retraining,
  i.e. only if features must be rebuilt or new audio is added.
- **Paid GPU VM / other HPC:** clone from GitHub, restore the critical
  archive, install dependencies, and use the GPU only for re-extraction.

## 10. Tracker policy after this checkpoint

The science state is held: `current_task: D4.3`, `last_completed_task: D4.2`,
`current_cut: D4`, `current_phase: 4`, `blocked: true`. A separate top-level
`migration:` section records this checkpoint. The EXIT.* tasks are not added to
the science task history.

## 11. Exit tag

After the migration commit and once archive checksums exist, an annotated tag
`dj-clustering-surrey-exit-2026-05-31` marks the checkpoint and embeds the
critical-archive sha256.
