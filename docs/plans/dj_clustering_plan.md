# DJ Music Collection Clustering: Deterministic Implementation Plan (v6)

## 0. How to use this plan

This file is the execution plan for the DJ music collection clustering project. It converts the strategy specification into a sequential implementation plan that Claude Code can follow without making ad-hoc design decisions.

Source-of-truth rule:

```text
This plan is the sole operational source of truth for this branch. If any external document conflicts with this plan, this plan wins unless the user explicitly approves a change.
```


Read this file after `CLAUDE.md` and before implementation work.

`CLAUDE.md` defines standing execution rules, repository policy, Git identity policy, coding rules, and scope guardrails. This file defines task order, task boundaries, decision rules, gates, verification criteria, progress tracker rules, and the exact technical defaults for the first implementation.

Execution rule:

1. Read `CLAUDE.md`.
2. Read `docs/plans/dj_clustering_plan.md`.
3. Read `docs/progress/dj_clustering_progress.md` and `docs/progress/dj_clustering_progress.yaml` if they exist.
4. Identify the first pending task.
5. Execute only that task.
6. Run the verification for that task.
7. Update the DJ clustering progress trackers.
8. Commit only source code, configs, documentation, small reports, and tracker updates.
9. Do not commit local music files, downloaded audio, embeddings, large pairwise tables, caches, model checkpoints, generated playlists with private paths unless explicitly allowed, or large run artifacts.
10. Stop at gates and report status.

If a task depends on an earlier task that is not complete, do the earlier task first and update the tracker.

Bootstrap rule:

```text
If docs/plans/dj_clustering_plan.md does not exist, restore it from git history on feature/dj-clustering-v1 before implementation work.
If docs/progress/dj_clustering_progress.md or docs/progress/dj_clustering_progress.yaml do not exist, create them in Task D0.3.
```

## 1. Branch role

This branch exists to build a deterministic music organization system for a local DJ collection.

The current product is not an ordered playlist generator, transition predictor, cue-point system, or automatic DJ mixing system.

Working branch:

```text
feature/dj-clustering-v1
```

Integration branch:

```text
main
```

Existing reference material:

```text
Current V4 implementation:
  src/v4/
  tests/v4/
  config/v4.yaml
  slurm/jobs/v4/
  docs/v4/
  playlists/V4_*/
  requirements_v4.txt

Legacy implementations:
  legacy/v1/
  legacy/v2/
  legacy/v3/
```

Rules:

1. Create `feature/dj-clustering-v1` from the current `main` branch unless another active DJ branch already exists.
2. Do not merge directly into `main` without a pull request or explicit user instruction.
3. Treat V2 and current V4 as diagnostic references or baselines, not as ground truth.
4. Do not move current V4 to `legacy/` during D0-D7.
5. Do not rewrite, delete, or reorganize `src/v4/`, `tests/v4/`, `config/v4.yaml`, `slurm/jobs/v4/`, `docs/v4/`, or `playlists/V4_*/` during D0-D7 unless a later explicit cleanup task is approved by the user.
6. Build the new deterministic pipeline in `src/dj_clustering/`, `scripts/dj_clustering/`, `configs/dj_clustering/`, `tests/dj_clustering/`, and `reports/dj_clustering/`.
7. Reuse or adapt current V4 code only after the D0.5 reuse audit classifies the component and records the reason.
8. Do not commit private audio or generated large artifacts.
9. Commit and push automatically after successful task completion using Gabriel Bibbó <gabobibbo@gmail.com> as author.
10. Do not add Co-Authored-By, Generated-By, AI-authorship, or assistant attribution trailers.

Recommended Git workflow:

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout feature/dj-clustering-v1 || git checkout -b feature/dj-clustering-v1
git push -u origin feature/dj-clustering-v1
```

Decision rules:

1. If the branch already exists locally, fetch and continue there after checking status.
2. If the branch exists remotely but not locally, check it out from origin.
3. If uncommitted user changes exist, stop and report before editing files.
4. If merge conflicts touch legacy outputs or private path manifests, stop and report before resolving.
5. If only progress trackers conflict, preserve both histories and update current state manually.

## 2. Project goal

The goal is to organize a local electronic music collection into musically useful DJ folders and overlapping clusters.

The branch must produce:

1. a reproducible local inventory of the user's decoded music library;
2. stable track identifiers that survive path changes and duplicate detection;
3. feature extraction scripts for frozen pretrained representations and metadata;
4. manual triplet question queues for low-effort user feedback;
5. optional weak evidence from 1001Tracklists with explicit confidence and splits;
6. pairwise feature tables and similarity profiles;
7. deterministic sweep and ranking evaluation;
8. overlapping cluster assignment with primary and auxiliary memberships;
9. bridge track detection;
10. export manifests and optional M3U playlists without duplicating audio by default;
11. small review reports suitable for Git;
12. current progress trackers.

The branch must support three possible outcomes:

1. a usable frozen-embedding clustering system that passes minimum success criteria;
2. a frozen-embedding system that is usable but marked limited and needs more user labels or more audio coverage;
3. a Regime 2 projection-head path if Regime 1 fails and evidence thresholds are met.

The branch must not claim that broad genre classification is solved. The relevant target is DJ-relevant folder coherence.

## 3. Relationship to existing project versions

The current repository contains one active V4 implementation and older legacy implementations.

Current V4 is not legacy for this plan. It is the current implementation baseline and reference code:

```text
src/v4/
tests/v4/
config/v4.yaml
slurm/jobs/v4/
docs/v4/
playlists/V4_*/
requirements_v4.txt
```

Older legacy implementations are reference material:

```text
legacy/v1/
legacy/v2/
legacy/v3/
```

Legacy V2 appears to have produced better subjective organization than the latest exported V4 playlists. This is a hypothesis to evaluate, not a target to imitate.

Current V4 appears to include MERT-based logic, PCA or dimensionality reduction, HDBSCAN, optional noise reassignment, evaluation code, UI code, Slurm jobs, configs, docs, and exported M3U playlists. These choices must be audited and evaluated, not assumed correct and not overwritten in place.

V4 status vocabulary:

```text
v4_code_available:
  V4 source code, tests, config, docs, Slurm jobs, and requirements exist.

v4_artifacts_available:
  reusable heavy artifacts such as embeddings, track_uids, clustering parquet files, or similarity artifacts exist locally.

v4_playlist_only:
  exported V4 M3U playlists exist, but reusable heavy artifacts are absent.

v4_competing:
  V4 can provide a reusable embedding or similarity space and can compete in the main composite score.

v4_diagnostic_only:
  V4 can provide cluster labels, playlists, reports, or implementation ideas, but cannot provide a fair similarity space.
```

Rules:

1. V2 may enter the leaderboard only if reusable embeddings or pairwise similarity can be loaded or reproduced cheaply.
2. If V2 has only cluster labels or M3U exports, it is diagnostic-only and cannot compete for the main composite score.
3. Current V4 must remain in place during D0-D7.
4. Current V4 must be audited in D0.5 before any reuse or adaptation.
5. If V4 code exists but heavy artifacts are absent, record `v4_code_available` and classify the leaderboard role as `v4_diagnostic_only` unless artifacts are found later.
6. If V4 playlists exist but cannot be mapped to current `track_id` values with adequate coverage, use preliminary clustering rather than V4 playlists for triplet boundary generation.
7. Noise reassignment must be treated as a hypothesis and compared against explicit alternatives.
8. No V2 or V4 result should be used as ground truth for training or model selection.
9. Manual triplets, validation evidence, and operational guardrails decide model selection.
10. New generated M3U files with private absolute paths must be written under ignored artifact roots unless the user explicitly approves committing sanitized versions.
11. Existing `playlists/V4_*` files must not be modified or deleted during D0-D7. Treat them as reference exports.

## 4. Repository location and plan placement

The plan must live inside the repository, not beside it.

Required plan file:

```text
docs/plans/dj_clustering_plan.md
```

Required progress trackers:

```text
docs/progress/dj_clustering_progress.md
docs/progress/dj_clustering_progress.yaml
```

Recommended runtime roots outside Git:

```text
runs/dj_clustering/
artifacts/dj_clustering/
.cache/
data/local_music_inventory/
```

These paths may exist under the repository only if ignored by Git. Private audio must not be committed.

Decision rules:

1. If the repo root is unknown, run `git rev-parse --show-toplevel` and use that path.
2. If the checked-out repo already contains `docs/plans/`, reuse it.
3. If both a local Windows copy and an HPC or server copy exist, GitHub is the source of synchronization for code and small reports.
4. Do not manually copy edited source files between machines unless Git is unavailable.
5. Large artifacts live outside Git and are referenced by path, checksum, run ID, or manifest ID.

## 5. Fixed technical decisions

These decisions are closed for the first implementation.

1. Primary product: folders and overlapping clusters.
2. Non-goals for now: ordered playlist continuation, transition prediction, cue point analysis, automatic mixing, and automatic DJ set sequencing.
3. Similarity target: DJ-relevant folder coherence, not broad genre classification.
4. Main manual evidence: two-candidate triplets with skip.
5. Manual triplets are relative comparisons, not absolute negatives.
6. Current playlist evidence: do not rely on existing Traktor playlists because the user does not trust them.
7. 1001Tracklists role: weak evidence and evaluation support, not ground truth and not a clustering feature in Regime 1.
8. External genre datasets role: secondary sanity checks only.
9. First modeling regime: frozen pretrained embeddings plus structured search.
10. Second modeling regime: projection head over frozen embeddings only if Regime 1 fails and evidence thresholds are met.
11. Third modeling regime: full encoder fine-tuning is out of scope for this plan.
12. Track identity: separate `audio_content_hash` from unique `track_id`.
13. Exact duplicate audio files: keep duplicate rows in inventory, but exclude duplicate copies from clustering by default.
14. Decode failures: keep rows in inventory with `decode_status = failed`, but exclude them from features, pair tables, triplets, clustering, and export.
15. Primary clustering: one primary cluster per canonical decoded track.
16. Auxiliary clustering: zero to two auxiliary clusters per canonical decoded track.
17. Export default: manifests and M3U playlists, not duplicated audio files.
18. Reproducibility: fixed seeds and recorded effective composite weights.
19. Metrics must be reported with component breakdowns, not only a single composite number.
20. No private audio, downloaded audio, embeddings, large pairwise matrices, model checkpoints, or generated audio in Git.

## 6. Target repository layout

Use this layout for DJ clustering files:

```text
traktor-main/
  docs/
    plans/
      dj_clustering_plan.md
    progress/
      dj_clustering_progress.md
      dj_clustering_progress.yaml
    dj_clustering/
      methodology_notes.md
  configs/
    dj_clustering/
      inventory.yaml
      features.yaml
      sweep_regime1.yaml
      export.yaml
  scripts/
    dj_clustering/
      inventory_library.py
      extract_features.py
      build_pair_features.py
      generate_triplet_questions.py
      ingest_manual_triplets.py
      match_1001tracklists.py
      run_similarity_sweep.py
      pick_winner.py
      assign_overlapping_clusters.py
      export_cluster_manifests.py
      summarize_run.py
  src/
    dj_clustering/
      inventory.py
      identity.py
      features.py
      pair_features.py
      similarity_profiles.py
      triplets.py
      evidence.py
      sweep.py
      clustering.py
      export.py
      metrics.py
  tests/
    dj_clustering/
      test_identity.py
      test_pair_features.py
      test_triplets.py
      test_similarity_profiles.py
      test_clustering_membership.py
      test_export_manifests.py
  reports/
    dj_clustering/
  runs/
    dj_clustering/              # ignored
  artifacts/
    dj_clustering/              # ignored
```

Runtime-only paths, ignored by Git:

```text
runs/
artifacts/
data/
.cache/
*.wav
*.flac
*.mp3
*.m4a
*.aiff
*.aif
*.ogg
*.npy
*.npz
*.parquet
*.pt
*.pth
*.ckpt
*.onnx
```

Decision rules:

1. If compatible directories already exist, reuse them.
2. If a script belongs to the DJ clustering workflow, put it under `scripts/dj_clustering/`.
3. If reusable implementation logic is needed, put it under `src/dj_clustering/`.
4. If a report is small and useful for GitHub, put it under `reports/dj_clustering/`.
5. If an artifact contains private paths, large pairwise data, embeddings, or audio, keep it outside Git and reference it from a small report.
6. The approved implementation namespace for this branch is `src/dj_clustering/` and `scripts/dj_clustering/`; this does not violate no-root-pollution rules because current V4 must remain frozen.
7. Do not refactor `src/v4/` in place unless a later task explicitly requests it and the user approves the cleanup scope.
8. If `CLAUDE.md` prefers refactoring existing code over adding scripts, this plan authorizes the separate DJ clustering namespace because V4 is a frozen baseline/reference for D0-D7. Record the exception in `docs/PROJECT_MAP.md` during D0.5.

## 7. Implementation cuts

Implementation is split into eight cuts, D0 to D7. A later cut must not start before the previous cut gate passes, except where explicitly marked as parallel setup.

### Cut D0. Plan, branch, and tracker bootstrap

Immediate target: make the branch safe, install the plan, create trackers, and protect private artifacts from Git.

### Cut D1. Inventory and identity foundation

Target: create a reproducible local music inventory with stable IDs, decode status, exact duplicate detection, and canonical track selection.

### Cut D2. Feature foundation and reference availability

Target: extract frozen audio and metadata features, validate feature availability, and classify legacy V2 plus current V4 references as competing, diagnostic-only, playlist-only, code-available, or unavailable.

### Cut D3. Pair features and evaluation evidence

Target: build pairwise feature tables, create manual triplet evidence, and optionally create 1001Tracklists weak evidence.

### Cut D4. First sweep and active manual evidence

Target: run an exploratory sweep, generate active triplet questions from candidate disagreement, and collect enough non-skip triplets for the full sweep.

### Cut D5. Full sweep and winner selection

Target: freeze evidence splits, run the full Regime 1 sweep, compute validation metrics, and select a winner without test leakage.

### Cut D6. Overlapping clustering and export

Target: assign primary and auxiliary clusters, compute bridge scores, export manifests and M3U files, and run manual sample review.

### Cut D7. Final evaluation, fallback, and optional Regime 2

Target: report held-out test results only after the winner is locked, try alternative winners if validation or user review fails, collect more evidence if needed, and run Regime 2 only if required and evidence thresholds are met.

## 8. Progress tracker format

Maintain both DJ clustering trackers.

YAML shape:

```yaml
branch: feature/dj-clustering-v1
integration_branch: main
current_cut: D0
current_phase: 0
current_task: "D0.1"
last_completed_task: null
blocked: false
blocker: null
repo_root: null
library_root: null
artifact_root: null
current_run_id: null
regime: 1
v2_status: unknown
v4_status: unknown
v4_code_status: unknown
v4_artifact_status: unknown
v4_playlist_status: unknown
manual_triplets_answered: 0
validated_1001_positives: 0
tasks:
  "D0.1": pending
  "D0.2": pending
  "D0.3": pending
  "D0.4": pending
  "D0.5": pending
```

Markdown shape:

```markdown
# DJ Clustering Task Progress

Branch: feature/dj-clustering-v1
Integration branch: main
Current cut: D0
Current phase: Phase 0
Current task: Task D0.1

## Completed

None yet.

## Current blocker

None.

## Current run

None yet.

## Next task

Task D0.1. Inspect repository state.
```

After every task:

1. update completed task;
2. update next task;
3. record commands run;
4. record tests or checks run;
5. record output artifact paths;
6. record failures or unverified parts;
7. record run ID if one was created;
8. record whether generated artifacts are tracked or ignored;
9. do not mark a gate complete without passing checks.

Decision rules:

1. If a task passes verification, mark it complete.
2. If a task fails verification, keep it current and record failure.
3. If a task cannot run because of a missing dependency or missing user input, mark `blocked: true` with a concrete blocker.
4. If a gate passes, stop and report.
5. If a gate fails, do not advance.

## 9. Phase 0. Branch, plan, and tracker bootstrap

Goal: make the DJ clustering work safe to run and track independently.

### Task D0.1. Inspect repository state

Actions:

1. locate the repository root;
2. run `git rev-parse --show-toplevel`;
3. inspect current branch;
4. inspect Git remote;
5. inspect uncommitted changes;
6. inspect whether `docs/plans/`, `docs/progress/`, `src/`, `scripts/`, `reports/`, and `tests/` exist;
7. inspect current V4 locations if present: `src/v4/`, `tests/v4/`, `config/v4.yaml`, `slurm/jobs/v4/`, `docs/v4/`, `docs/V4_USAGE.md`, `playlists/V4_*/`, and `requirements_v4.txt`;
8. inspect legacy folders relevant to V1, V2, and V3;
9. do not delete existing code or artifacts.

Suggested commands:

```bash
pwd
git rev-parse --show-toplevel
git status --short
git branch --show-current
git remote -v
ls -la
ls -la docs || true
ls -la legacy || true
ls -la src || true
ls -la src/v4 || true
ls -la tests/v4 || true
ls -la config || true
ls -la slurm/jobs/v4 || true
ls -la docs/v4 || true
ls -la docs/V4_USAGE.md || true
ls -la playlists || true
ls -la scripts || true
```

Done when:

1. repo root is recorded;
2. active branch is recorded;
3. remote is recorded;
4. dirty working tree status is recorded;
5. relevant current V4 locations are recorded;
6. relevant legacy folders are recorded;
7. next task can be applied to the actual checkout.

Decision rules:

1. If this is not a Git repo, stop and report the exact path.
2. If the repo has uncommitted user changes, stop and report before changing files.
3. If the remote is not the expected project remote, record the mismatch and stop.
4. If the repo is already on `feature/dj-clustering-v1`, continue after fetching.

### Task D0.2. Create or sync the DJ clustering branch

Actions:

1. fetch GitHub state;
2. confirm `main` exists locally or remotely;
3. create `feature/dj-clustering-v1` from `main` if missing;
4. otherwise sync the existing branch with `origin/main` if safe;
5. push the branch if it is new;
6. configure Git author as Gabriel Bibbó <gabobibbo@gmail.com> if not already configured.

Suggested commands:

```bash
git fetch origin
git checkout main || git checkout -b main origin/main
git pull --ff-only origin main
git checkout feature/dj-clustering-v1 || git checkout -b feature/dj-clustering-v1 origin/feature/dj-clustering-v1 || git checkout -b feature/dj-clustering-v1
git config user.name "Gabriel Bibbó"
git config user.email "gabobibbo@gmail.com"
git push -u origin feature/dj-clustering-v1
```

Done when:

1. DJ clustering branch exists;
2. branch tracks `origin/feature/dj-clustering-v1`;
3. current branch is `feature/dj-clustering-v1`;
4. Git author is correct;
5. tracker records branch state.

Decision rules:

1. If branch creation fails because the branch exists remotely, check it out from origin.
2. If branch is behind `main`, merge or rebase only if the working tree is clean.
3. If conflicts occur, stop and report.
4. Do not merge into `main` in this task.

### Task D0.3. Install plans and create independent trackers

Actions:

1. create `docs/plans/` if missing;
2. create or update `docs/plans/dj_clustering_plan.md`;
3. verify plan file exists before implementation work continues;
4. create `docs/progress/` if missing;
5. create `docs/progress/dj_clustering_progress.md` if missing;
6. create `docs/progress/dj_clustering_progress.yaml` if missing;
7. initialize current cut as `D0`;
8. initialize current task as the first incomplete task;
9. do not reuse unrelated legacy trackers for this branch.

Done when:

1. `docs/plans/dj_clustering_plan.md` exists inside the repo;
2. both DJ clustering trackers exist;
3. tracker format is valid;
4. next pending task is visible;
5. tracker does not conflict with unrelated branch trackers.

Decision rules:

1. If trackers already exist, update them without deleting history.
2. If Markdown and YAML trackers disagree, use Markdown for history and YAML for current state.
3. If tracker current task is earlier than completed history, correct YAML and record the correction.

### Task D0.4. Configure ignored runtime artifact paths

Actions:

1. inspect `.gitignore`;
2. ensure runtime artifacts are ignored;
3. add ignore rules for local music data, embeddings, pairwise tables, caches, run folders, model artifacts, and generated audio;
4. do not ignore source code, configs, tests, small reports, or tracker files.

Required ignore patterns:

```text
runs/
artifacts/
data/
.cache/
*.wav
*.flac
*.mp3
*.m4a
*.aiff
*.aif
*.ogg
*.npy
*.npz
*.parquet
*.pt
*.pth
*.ckpt
*.onnx
```

Done when:

1. heavy and private outputs are ignored;
2. configs, scripts, tests, docs, and small reports remain trackable;
3. `.gitignore` contains no secrets;
4. tracker records the change.

Decision rules:

1. If `.gitignore` already contains equivalent rules, do not duplicate them.
2. If a report directory is ignored accidentally, narrow the rule.
3. If a required artifact is too large for GitHub, keep it ignored and document the external location.

### Task D0.5. Audit current V4 implementation and create reuse map

Actions:

1. inspect current V4 files and directories if present:
   - `src/v4/`
   - `tests/v4/`
   - `config/v4.yaml`
   - `slurm/jobs/v4/`
   - `docs/v4/`
   - `docs/V4_USAGE.md`
   - `playlists/V4_*/`
   - `requirements_v4.txt`
2. inspect legacy folders if present:
   - `legacy/v1/`
   - `legacy/v2/`
   - `legacy/v3/`
3. classify each relevant V4 component as one of:
   - `reuse_by_import`
   - `adapt_into_src_dj_clustering`
   - `diagnostic_reference_only`
   - `out_of_scope_for_v1`
4. do not move, delete, rename, or rewrite V4 files.
5. write `reports/dj_clustering/v4_reuse_map.md`.
6. update `docs/PROJECT_MAP.md` with the DJ clustering plan location, tracker location, approved namespace, and frozen V4 policy.
7. record whether V4 is currently `v4_code_available`, `v4_artifacts_available`, `v4_playlist_only`, `v4_competing`, and/or `v4_diagnostic_only`.

Recommended initial classification:

```text
reuse_by_import or adapt_into_src_dj_clustering:
  src/v4/common/catalog.py
  src/v4/common/audio_utils.py
  src/v4/common/embedding_utils.py
  src/v4/common/demucs_utils.py
  src/v4/evaluation/metrics.py

diagnostic_reference_only:
  src/v4/pipeline/phase4_order.py
  src/v4/pipeline/phase5_export.py
  src/v4/ui/app.py
  playlists/V4_*/
  docs/v4/
  docs/V4_USAGE.md
  slurm/jobs/v4/
  config/v4.yaml

out_of_scope_for_v1:
  transition-aware ordering
  Streamlit UI changes
  automatic playlist sequencing
  modifying or relocating current V4 files
```

Done when:

1. `reports/dj_clustering/v4_reuse_map.md` exists;
2. `docs/PROJECT_MAP.md` records the new DJ clustering namespace and frozen V4 policy;
3. current V4 remains untouched;
4. every reused or adapted V4 component has an explicit reason;
5. tracker records whether V4 can be used as `v4_competing`, `v4_diagnostic_only`, `v4_playlist_only`, or only `v4_code_available`.

Decision rules:

1. If a recommended V4 file does not exist, record it as absent and continue.
2. If V4 heavy artifacts are absent from Git, mark `v4_artifacts_available: false` and do not try to reconstruct them in D0.
3. If only V4 playlists exist, mark `v4_playlist_only` and defer mapping to D3.2.
4. If `CLAUDE.md` conflicts with the separate namespace decision, follow this plan for the namespace and record the exception in `docs/PROJECT_MAP.md`.
5. If any V4 file is accidentally modified, revert that change before completing the task unless the user explicitly approved it.

Cut D0 gate: stop after D0.5 and report branch and tracker readiness.

## 10. Phase 1. Inventory and identity foundation

Goal: create a reproducible inventory of the local collection, without extracting expensive embeddings yet.

### Task D1.1. Define local library configuration

Actions:

1. create `configs/dj_clustering/inventory.yaml`;
2. define one or more library roots outside Git;
3. define allowed audio extensions;
4. define output root for inventory artifacts;
5. define whether recursive scanning is enabled;
6. define whether hidden files are ignored;
7. define path handling policy for reports and M3U output.

Required defaults:

```yaml
allowed_extensions:
  - .wav
  - .flac
  - .mp3
  - .m4a
  - .aiff
  - .aif
  - .ogg
recursive: true
ignore_hidden_files: true
m3u_paths: absolute
hash_method: sha256_full_content
```

Done when:

1. inventory config exists;
2. library root is explicit or marked pending user path;
3. artifact output root is ignored by Git;
4. config can be parsed.

Decision rules:

1. If the library root is unknown, create a placeholder config and block before D1.2.
2. If the configured root is inside the Git repo, stop and ask whether this is intentional.
3. If paths contain private user information, keep full paths only in ignored artifacts and summarize sanitized paths in committed reports.

### Task D1.2. Implement library inventory script

Actions:

1. implement `scripts/dj_clustering/inventory_library.py`;
2. implement or update `src/dj_clustering/inventory.py` and `src/dj_clustering/identity.py`;
3. scan configured audio files;
4. read file size, extension, mtime, and path;
5. attempt metadata extraction for artist, title, album, genre, label, year, BPM, key, and folder hints;
6. attempt audio decode and duration read;
7. compute `audio_content_hash` as SHA256 of file content for decoded and undecoded audio when readable;
8. create unique `track_id` values from hash plus deterministic duplicate suffix where needed;
9. mark exact duplicates by `duplicate_of_track_id`;
10. select one canonical representative per `audio_content_hash` for clustering;
11. record decode failures without blocking the full inventory.

Required output columns:

```text
track_id
audio_content_hash
canonical_track_id
is_canonical
is_exact_duplicate
duplicate_of_track_id
file_path
file_name
extension
file_size_bytes
duration_seconds
decode_status
decode_error
artist
title
album
genre
label
year
bpm
key
folder_path
folder_hint
metadata_quality_flags
hash_method
```

Done when:

1. `library_inventory.csv` exists in the ignored artifact root;
2. `library_inventory_summary.md` exists under `reports/dj_clustering/`;
3. `decode_status` is recorded for every row;
4. all `track_id` values are unique;
5. exact duplicate groups are reported;
6. canonical decoded track count is recorded as `N_canonical_decoded`;
7. tests for identity logic pass.

Decision rules:

1. If an audio file cannot be decoded, keep it in inventory with `decode_status = failed` and exclude it from feature extraction, pair features, triplet queues, clustering, and export.
2. If two files have the same `audio_content_hash`, keep the same hash but assign distinct `track_id` values using a deterministic duplicate suffix.
3. Exact duplicate audio files are excluded from clustering by default after selecting one canonical representative per `audio_content_hash`.
4. Duplicate rows remain in `library_inventory.csv` and are restored only at export time according to the canonical track assignment.
5. If more than 20 percent of files fail decode, stop after the inventory report and ask for review. This is an implementation guardrail, not a model-quality criterion.

### Task D1.3. Create metadata quality report

Actions:

1. summarize metadata coverage;
2. report class balance for genre, folder, artist, label, BPM, and key;
3. identify whether metadata sanity scores are valid;
4. report duplicate counts and decode failure counts;
5. recommend whether more music is needed before expensive embedding extraction.

Metadata sanity validity rule:

```text
A metadata field is valid for metadata_sanity_score only if it has at least 2 classes and no single class covers more than 85 percent of canonical decoded tracks.
```

Done when:

1. `reports/dj_clustering/metadata_quality_report.md` exists;
2. valid and invalid metadata fields are listed;
3. duplicate and decode failure counts are listed;
4. `N_canonical_decoded` is recorded;
5. tracker records whether the project can proceed to feature extraction.

Decision rules:

1. If `N_canonical_decoded < 100`, proceed only with a limited pilot flag.
2. If `100 <= N_canonical_decoded < 500`, use small-N thresholds in this plan.
3. If `N_canonical_decoded >= 500`, use standard thresholds and record that the dataset is no longer small-N.
4. If metadata has no valid sanity field, set `metadata_sanity_score_unavailable = true` and renormalize composite weights later.

Cut D1 gate: inventory, identity, duplicate policy, decode status, and metadata quality report must be complete before feature extraction.

## 11. Phase 2. Feature extraction foundation

Goal: compute track-level frozen features for canonical decoded tracks only.

### Task D2.1. Create feature extraction config

Actions:

1. create `configs/dj_clustering/features.yaml`;
2. define audio loading sample rates;
3. define segment policy;
4. define MERT aggregation choices;
5. define whether HPSS or Demucs is used for percussion;
6. define output artifact paths;
7. define fallback behavior when optional libraries are unavailable.

Committed segment policy:

```text
If duration >= 150s:
  use three 30s non-overlapping segments centered near 25%, 50%, and 75%, avoiding first and last 30s.

If 75s <= duration < 150s:
  use one middle 60s segment.

If 35s <= duration < 75s:
  use one centered 30s segment.

If duration < 35s:
  use the full track and mark short_track = true.
```

Committed MERT percussion policy:

```text
mert_perc is computed from HPSS-percussive audio by default.
If Demucs stems already exist, they may be used as a higher-quality percussion source.
If neither HPSS nor Demucs is available, omit mert_perc and all dependent configs, including mert_concat.
```

Committed concatenation policy:

```text
mert_concat = concatenate L2-normalized mert_full and L2-normalized mert_perc track-level embeddings, then L2-normalize the concatenated vector.
```

Done when:

1. feature config exists;
2. segment policy is represented in config or code constants;
3. output paths are ignored by Git;
4. optional feature availability policy is explicit.

Decision rules:

1. If no MERT-capable environment is available, stop and record blocker.
2. If Essentia is unavailable, continue without Essentia and record absence.
3. If HPSS fails for a track, omit `mert_perc` for that track and record the failure.
4. If too many tracks fail a feature source, omit that feature source from the sweep and record why.

### Task D2.2. Implement feature extraction

Actions:

1. implement `scripts/dj_clustering/extract_features.py`;
2. implement or update `src/dj_clustering/features.py`;
3. load only canonical decoded tracks;
4. generate `segment_manifest.csv`;
5. compute MERT full-mix embeddings;
6. compute MERT percussion embeddings when available;
7. compute `mert_concat` when both full and percussion embeddings exist;
8. compute optional Essentia/EffNet features when available;
9. compute metadata-derived numeric features where applicable;
10. L2-normalize track-level embeddings before storage;
11. write feature quality report.

Feature aggregation rule:

```text
Compute segment embeddings.
L2-normalize each segment embedding.
Average segment embeddings per track.
L2-normalize the resulting track-level embedding.
```

Supported MERT aggregation values for Regime 1:

```text
last_layer_mean
last_4_layers_mean
layer_7_mean
```

Done when:

1. feature arrays exist outside Git;
2. `feature_manifest.csv` exists outside Git;
3. `segment_manifest.csv` exists outside Git;
4. `feature_quality_report.md` exists under `reports/dj_clustering/`;
5. every canonical decoded track is either embedded or has an explicit feature failure reason;
6. tests for segment selection and `mert_concat` normalization pass.

Decision rules:

1. If feature extraction fails for more than 10 percent of canonical decoded tracks for the default embedding source, stop and investigate. This is an implementation guardrail, not a model-quality criterion.
2. If a non-default feature source fails, exclude dependent configs and continue.
3. If generated arrays are too large, keep them ignored and commit only summaries.
4. If the feature script accidentally stages arrays or audio, unstage them and fix `.gitignore`.

### Task D2.3. Evaluate V2 and current V4 availability

Actions:

1. inspect legacy V2 artifacts;
2. inspect current V4 code, configs, tests, docs, Slurm jobs, and playlists as recorded in D0.5;
3. inspect whether current V4 heavy artifacts are present locally, including embeddings, track UIDs, clustering parquet files, pairwise tables, or similarity artifacts;
4. determine whether V2 can provide embeddings, pairwise similarity, cluster labels, or only playlist exports;
5. classify V2 as `competing`, `diagnostic_only`, or `unavailable`;
6. classify current V4 using the status vocabulary from Section 3: `v4_code_available`, `v4_artifacts_available`, `v4_playlist_only`, `v4_competing`, and/or `v4_diagnostic_only`;
7. write `reports/dj_clustering/reference_availability_report.md`.

Done when:

1. V2 status is recorded;
2. current V4 code status is recorded;
3. current V4 artifact status is recorded;
4. current V4 playlist status is recorded;
5. required files for each status are listed;
6. tracker records the selected status values.

Decision rules:

1. If V2 provides only cluster labels and no reusable embedding or pairwise similarity, mark `diagnostic_only`.
2. If V2-compatible embeddings can be loaded or recomputed in under 6 GPU hours, mark `competing`.
3. If current V4 provides reusable embeddings or a fair similarity space, mark `v4_competing`.
4. If current V4 provides code, configs, tests, docs, or Slurm jobs but no reusable heavy artifacts, mark `v4_code_available` and `v4_diagnostic_only`.
5. If only current V4 M3U playlists are available, mark `v4_playlist_only` and `v4_diagnostic_only`.
6. If a reference artifact cannot be parsed, record it and do not block the main pipeline.
7. Do not move, rewrite, or regenerate V4 artifacts in this task.

Cut D2 gate: default features, feature manifest, feature quality report, and reference availability report must be complete before pair features and evidence generation.

## 12. Phase 3. Pair features and evaluation evidence

Goal: build pairwise feature inputs, create manual triplet queues, and optionally add 1001Tracklists weak evidence.

### Task D3.1. Build pairwise feature table

Actions:

1. implement `scripts/dj_clustering/build_pair_features.py`;
2. implement or update `src/dj_clustering/pair_features.py` and `src/dj_clustering/similarity_profiles.py`;
3. compute pair features over canonical decoded tracks;
4. for `N_canonical_decoded <= 2000`, compute all unordered pairs;
5. for larger collections, compute all manual and weak-supervision pairs, plus top-100 nearest-neighbor pairs per representation and additional cross-cluster boundary pairs for bridge detection;
6. normalize all similarity features to `[0,1]`;
7. write `pair_features.parquet` outside Git;
8. write `pair_features_summary.md` under `reports/dj_clustering/`.

Pair policy:

```text
If N_canonical_decoded <= 2000:
  compute all unordered canonical decoded pairs.

If N_canonical_decoded > 2000:
  compute all manual and weak-supervision pairs,
  plus top-100 nearest-neighbor pairs per representation,
  plus additional cross-cluster boundary pairs for bridge detection.
```

Similarity normalization:

```text
cosine_embedding_01 = (cosine_embedding + 1) / 2
similarity = weighted sum of features scaled to [0,1]
distance = 1 - similarity
```

Metadata similarity rule:

```text
Missing metadata fields are excluded from the denominator.
If no metadata fields are available for a pair, metadata_similarity = 0 and metadata_similarity_available = false.
```

BPM similarity rule:

```text
Compare BPM under 0.5x, 1x, and 2x tempo equivalence.
Use the smallest effective difference.
```

Committed similarity profiles:

```text
audio_only:
  1.00 * cosine_embedding_01

audio_plus_bpm_key_light:
  0.90 * cosine_embedding_01
  0.05 * bpm_similarity
  0.05 * key_compat

audio_plus_metadata_light:
  0.85 * cosine_embedding_01
  0.15 * metadata_similarity

audio_plus_bpm_key_metadata_light:
  0.80 * cosine_embedding_01
  0.05 * bpm_similarity
  0.05 * key_compat
  0.10 * metadata_similarity
```

Regime 1 restriction:

```text
1001Tracklists evidence must not be used as a clustering feature in Regime 1.
It may be used only for evaluation or later Regime 2 training, according to split policy.
```

Done when:

1. `pair_features.parquet` exists outside Git;
2. pair count matches policy;
3. all similarity profile columns are scaled to `[0,1]`;
4. `pair_features_summary.md` exists;
5. tests for scaling and metadata missingness pass.

Decision rules:

1. If `pair_features.parquet` is too large, keep it ignored and commit only the summary.
2. If a feature source is missing, omit only the dependent profile and record it.
3. If pair features cannot be computed for the default embedding source, return to D2.

### Task D3.2. Generate initial manual triplet question queue

Actions:

1. implement `scripts/dj_clustering/generate_triplet_questions.py`;
2. implement or update `src/dj_clustering/triplets.py`;
3. generate the first 40 questions;
4. write `triplet_question_queue.csv` outside Git or under a sanitized review folder;
5. write optional `triplet_question_queue.m3u` with absolute paths by default;
6. write a small summary report;
7. if V4 playlists are used, write `reports/dj_clustering/v4_playlist_mapping_report.md`.

Initial 40-question policy:

```text
20 questions:
  anchor random among canonical decoded tracks,
  candidate B and C sampled from top-50 nearest neighbors under the default embedding source.

20 questions:
  anchor random among canonical decoded tracks,
  candidate B and C sampled near boundaries of baseline V4 clusters.
```

V4 boundary source policy:

```text
If V4 reusable cluster labels or artifacts are available, use those as a diagnostic boundary source.
If V4 artifacts are unavailable but `playlists/V4_*` exists, parse the latest V4 playlist export as a diagnostic boundary source only after mapping entries to current `track_id` values.
Map V4 playlist entries to `track_id` after D1 inventory using:
  1. `audio_content_hash` if available,
  2. normalized filename if hash is unavailable,
  3. normalized artist-title as fallback.
Use V4 playlist boundaries for triplet question generation only if mapping coverage is at least 70 percent of playlist entries that appear to belong to the scanned collection.
If coverage is below 70 percent, do not use V4 playlists for triplet generation.
```

Fallback if V4 labels or mapped V4 playlist boundaries are unavailable:

```text
Use preliminary MERT_full_HDBSCAN_default clusters.
If preliminary clustering is unavailable, use metadata or folder boundaries.
```

Triplet de-duplication policy:

```text
Do not ask duplicate triplets.
Do not ask the same anchor with the same unordered candidate pair twice.
Do not ask a triplet where anchor, B, or C share the same audio_content_hash unless the purpose is explicit duplicate review.
```

Required queue columns:

```text
question_id
anchor_track_id
candidate_b_track_id
candidate_c_track_id
anchor_artist
anchor_title
candidate_b_artist
candidate_b_title
candidate_c_artist
candidate_c_title
anchor_file_path
candidate_b_file_path
candidate_c_file_path
selection_source
```

Done when:

1. `triplet_question_queue.csv` exists;
2. it contains 40 non-duplicate questions unless insufficient data is documented;
3. optional M3U uses absolute paths by default;
4. no question includes duplicate audio hashes unless explicitly marked;
5. summary report exists.

Decision rules:

1. If there are fewer than 40 valid questions, generate as many as possible and record limitation.
2. If no baseline clusters exist, use the fallback policy.
3. If paths are too private for Git, keep the queue ignored and commit only a summary.

### Task D3.3. Ingest manual triplet answers

Actions:

1. implement `scripts/dj_clustering/ingest_manual_triplets.py`;
2. accept answers in CSV form;
3. validate `question_id`, `anchor_track_id`, `candidate_b_track_id`, `candidate_c_track_id`, and answer;
4. allow only `B`, `C`, or `skip` as answer values;
5. write `manual_triplets.csv`;
6. write `manual_triplets_summary.md`.

Manual triplet semantics:

```text
If answer = B:
  evidence is d(anchor, B) < d(anchor, C).

If answer = C:
  evidence is d(anchor, C) < d(anchor, B).

If answer = skip:
  log the answer but do not convert it into positive or negative evidence.
```

Pair conversion rule:

```text
If triplets are converted to eval_pairs.csv,
mark the chosen pair as positive,
and mark the non-chosen pair as relative_negative.
The positive label must include source metadata indicating that it came from a relative triplet comparison.
Do not treat relative_negative as an absolute negative for projection-head training unless explicitly confirmed by the user.
```

Done when:

1. answered triplets are validated;
2. skipped questions are recorded but excluded from accuracy metrics;
3. no absolute negative labels are created from relative triplets;
4. summary reports answered count and skip rate.

Decision rules:

1. Until the first full sweep task D5.1, all answered triplets are exploration evidence.
2. The first full split is created only in D5.1.
3. If fewer than 20 non-skip answers exist, run only diagnostic sweeps and do not pick a winner.

### Task D3.4. Create 1001Tracklists matching input

Actions:

1. implement `scripts/dj_clustering/match_1001tracklists.py` in metadata-only mode first;
2. accept manually provided 1001Tracklists metadata exports, public setlist files, or scraper outputs if available;
3. normalize artist, title, version markers, remix/edit markers, and duration where available;
4. match against local canonical decoded tracks;
5. assign confidence scores;
6. write `1001_matches.csv` outside Git;
7. write `1001_matching_report.md`.

Identity validation policy:

```text
Prefer audio fingerprinting for identity validation.
If fingerprinting is unavailable, require normalized artist/title match plus duration difference <= 3 seconds plus compatible version/remix/edit markers.
MERT cosine may be stored as supporting evidence, but it must not be the only identity-acceptance rule.
```

Confidence policy:

```text
confidence = 1.0 if fingerprint or strong audio-identity evidence passes.
confidence = 0.7 if normalized artist/title + duration + version marker checks pass.
Reject matches below confidence 0.7.
```

1001 pair weighting policy:

```text
distance 1 in setlist: weight 1.0
distance 2 or 3 in setlist: weight 0.6
same set farther away: weight 0.2, diagnostic only unless explicitly enabled
same DJ or event context only: metadata context, not a positive pair by itself
```

Done when:

1. `1001_matches.csv` exists or the report states that no usable source was available;
2. accepted matches have confidence >= 0.7;
3. MERT cosine is not used as sole identity proof;
4. report includes accepted, rejected, and ambiguous counts.

Decision rules:

1. If fewer than 50 validated 1001 positives are available after splitting, use fallback composite later.
2. If no 1001 data is available, continue without it.
3. If downloaded audio is used for research, keep it outside Git and record restriction status in the report.
4. Do not block Regime 1 on 1001 data.

Cut D3 gate: pair features, initial manual triplet queue, manual ingestion path, and optional 1001 matching report must be complete before the first sweep.

## 13. Phase 4. Regime 1 first sweep and active triplets

Goal: run an initial diagnostic sweep and generate active questions from disagreement between candidate systems.

### Task D4.1. Implement Regime 1 sweep runner

Actions:

1. implement `scripts/dj_clustering/run_similarity_sweep.py`;
2. implement or update `src/dj_clustering/sweep.py`, `src/dj_clustering/metrics.py`, and `src/dj_clustering/clustering.py`;
3. implement committed grid with conditional branches;
4. include fixed baselines;
5. support diagnostic-only rows for V2 or V4;
6. write a leaderboard and component metrics;
7. write sweep config and run metadata.

Committed Regime 1 sweep grid:

```text
embedding_source:
  - mert_perc
  - mert_full
  - mert_concat

mert_aggregation:
  - last_layer_mean
  - last_4_layers_mean
  - layer_7_mean

normalization:
  - l2
  - none

dim_reduction:
  - none
  - pca_50
  - pca_100
  - umap_15

clusterer:
  - hdbscan
  - kmeans
  - agglomerative

hdbscan_min_cluster_size:
  - 4
  - 6
  - 8
  - 10

hdbscan_min_samples:
  - 1
  - 2

kmeans_k:
  - 10
  - 12
  - 15
  - 18

agglomerative_k:
  - 10
  - 12
  - 15
  - 18

noise_policy:
  - no_reassignment
  - confidence_limited_1nn
  - forced_1nn_export_only
```

Normalization policy:

```text
normalization = l2 is the preferred default.
normalization = none is retained only as a diagnostic ablation to detect whether preprocessing is harming downstream structure.
When a similarity profile uses cosine-based similarity, all cosine values are still converted to cosine_embedding_01 before profile weighting.
```

Conditional grid policy:

```text
For clusterer = hdbscan:
  use hdbscan_min_cluster_size
  use hdbscan_min_samples
  use noise_policy

For clusterer = kmeans:
  use kmeans_k
  set noise_policy = not_applicable
  run only on vector embeddings and vector projections
  do not use pairwise similarity profiles containing BPM, key, or metadata

For clusterer = agglomerative:
  use agglomerative_k
  set noise_policy = not_applicable
  use average linkage for precomputed distance matrices
  do not use ward linkage with precomputed distances
```

Fixed HDBSCAN baseline definition:

```text
Default HDBSCAN baseline:
  mert_aggregation = last_layer_mean
  segment_policy = default segment policy
  normalization = l2
  dim_reduction = umap_15
  similarity_profile = audio_only
  hdbscan_min_cluster_size = 6
  hdbscan_min_samples = 1
  noise_policy = no_reassignment

Default HDBSCAN baseline runs on the selected vector representation after dim_reduction = umap_15, using Euclidean distance in the UMAP space.
It does not use a precomputed pairwise distance matrix.
```

Required fixed baselines:

```text
current_V4 if reproducible and classified as v4_competing
V2_reproduced if available
MERT_perc_HDBSCAN_default
MERT_full_HDBSCAN_default
MERT_concat_HDBSCAN_default if mert_concat exists
```

Sweep cap policy:

```text
If the conditional grid exceeds 200 valid non-baseline configs, sample 200 non-baseline configs with random seed 13, then add all fixed baselines even if the final run count exceeds 200.
```

Noise policy definitions:

```text
no_reassignment:
  keep native HDBSCAN noise as unassigned for evaluation.

confidence_limited_1nn:
  reassign native noise only if nearest non-noise neighbor has cosine distance <= 0.30.

forced_1nn_export_only:
  keep native noise for metrics, but force an export assignment for manifest completeness.
```

Done when:

1. first sweep runner executes on available features;
2. invalid config combinations are skipped deterministically;
3. fixed baselines appear in the leaderboard if available;
4. diagnostic-only reference rows are marked clearly;
5. `first_sweep_leaderboard.csv` exists outside Git or as a small report if compact;
6. `reports/dj_clustering/first_sweep_summary.md` exists;
7. tests for conditional grid generation pass.

Decision rules:

1. If no manual triplets are available, run first sweep as diagnostic only.
2. If fewer than 20 non-skip triplets exist, do not select a winner.
3. If KMeans is selected, do not apply raw-noise penalties because raw noise is not applicable.
4. If HDBSCAN native soft membership is unavailable for precomputed distances, use medoid-distance fallback and record `membership_source = medoid_fallback`.
5. If agglomerative uses precomputed distances, use average linkage.

### Task D4.2. Generate active triplet questions

Actions:

1. compute disagreement among top candidate systems from first sweep;
2. generate additional triplet questions;
3. avoid duplicate questions and duplicate audio hashes;
4. write updated `triplet_question_queue.csv`;
5. write optional M3U review file;
6. write active selection summary.

Disagreement definition:

```text
For a candidate triplet (A, B, C), compare the top-3 configs in the leaderboard.
A disagreement exists if at least one config ranks B closer to A than C,
and at least one config ranks C closer to A than B.
```

Sorting metric:

```text
disagreement_margin = min(count_B_chosen, count_C_chosen) / total_top3_configs
Ties are broken by greater variance of the rank difference between B and C across configs.
```

Done when:

1. active question queue is generated;
2. queue avoids duplicates;
3. selection source is recorded per question;
4. summary explains how many questions came from disagreement.

Decision rules:

1. If first sweep did not produce at least two comparable configs, generate remaining questions from diverse anchors.
2. If too few disagreement cases exist, fill the queue with boundary cases.
3. Do not use held-out test labels because no held-out split exists yet.

### Task D4.3. Collect remaining triplet answers

Actions:

1. ingest additional user answers;
2. update `manual_triplets.csv`;
3. update manual summary;
4. record total non-skip answers.

Targets:

```text
Minimum before full sweep: 80 non-skip triplets.
Preferred before full Regime 1 sweep: 120 non-skip triplets.
Minimum before Regime 2: 200 non-skip triplets, or 100 non-skip triplets plus at least 1000 validated 1001 positives with confidence >= 0.7.
```

Done when:

1. at least 80 non-skip triplets exist, or limitation is recorded;
2. skipped questions are recorded;
3. tracker records manual triplet count.

Decision rules:

1. If fewer than 80 non-skip triplets exist, full sweep can run but cannot declare final success unless user explicitly accepts limited evidence.
2. If label quality appears inconsistent, create a report and ask for review before Regime 2.

Cut D4 gate: first sweep, active question generation, and at least 80 target triplets or a documented limitation must be complete before the full sweep.

## 14. Phase 5. Full sweep and winner selection

Goal: freeze splits, run the full Regime 1 sweep, and select a candidate without test leakage.

### Task D5.1. Create deterministic evidence splits and run full sweep

Actions:

1. create deterministic split assignments for manual triplets;
2. create deterministic split assignments for 1001 matches if present;
3. run full sweep using training and validation evidence only;
4. compute component metrics;
5. write full leaderboard;
6. write effective composite weights;
7. keep held-out test evidence unused for selection.

Manual split policy:

```text
At the first D5.1 run, create split_version = 1.
Freeze all existing manual triplet split assignments.
Use train / validation / held-out test.
The held-out test split is evaluated only once after the final selected configuration is locked.
```

If more triplets are collected later:

```text
Assign only new triplets deterministically to train or validation according to the current evidence goal.
Do not move existing held-out test triplets.
Do not use held-out test to trigger alternative winner selection.
```

1001 split policy:

```text
If 1001_matches.csv exists, create deterministic 1001 split assignments before the full sweep.
Store them in 1001_matches.csv and write 1001_split_manifest.json.
Only validation-split 1001 positives may enter recall_at_10_validated_1001_validation.
validated_1001_train may be used for Regime 2 training.
validated_1001_test, if created, is final-report-only.
```

Composite score:

```text
default_composite_score =
  0.45 * manual_triplet_accuracy_validation
+ 0.25 * recall_at_10_validated_1001_validation
+ 0.10 * metadata_sanity_score
+ 0.10 * bootstrap_stability_score
+ 0.10 * cluster_usability_score
- penalties
```

Fallback if 1001 validation positives are insufficient:

```text
Use fallback composite if validated 1001 positives in validation < 50.
For the first pilot, fallback composite is acceptable and expected.
```

Unavailable component rule:

```text
When a component is unavailable, remove it and divide each remaining component weight by the sum of remaining available weights.
Record the effective weights in the leaderboard.
```

Do not use held-out test in:

```text
composite score
winner selection
tie-breakers
alternative winner selection
Regime 2 hyperparameter selection
```

Done when:

1. manual split manifest exists;
2. 1001 split manifest exists if 1001 data exists;
3. full leaderboard exists;
4. effective weights are recorded;
5. held-out test metrics are not used for selection;
6. V2 row is marked `competing`, `diagnostic_only`, or `unavailable`.

Decision rules:

1. If fewer than 80 non-skip triplets exist, mark sweep as limited evidence.
2. If 1001 validation positives are fewer than 50, use fallback composite.
3. If metadata sanity is unavailable, renormalize weights.
4. If a config has raw_noise_rate = not_applicable, raw-noise penalties and raw-noise tie-breakers do not apply.

### Task D5.2. Compute metrics and guardrails

Actions:

1. compute manual triplet accuracy on validation split;
2. compute Recall@10 for validation 1001 positives when available;
3. compute metadata sanity when valid;
4. compute bootstrap stability;
5. compute cluster usability score;
6. compute penalties;
7. compute neighbor coherence@10;
8. compute bridge candidate quality score;
9. record all component metrics.

Bootstrap stability rule:

```text
Use 20 seeded 80 percent bootstrap subsamples.
For each subsample, cluster only the subsampled tracks and compare assignments against the full-data clustering restricted to those tracks.
Report ARI mean and std as primary stability metric.
Report NMI mean and std as secondary, not in composite.
Use clusterer seed = 13 for all bootstrap subsamples.
Total stability fits = 20 per config, not 5 * 20.
```

Neighbor coherence@10:

```text
k = 10
Weights by relation:
  manual positive: 1.0
  validated 1001 positive: 0.6
  same artist: 0.4
  same narrow genre: 0.3
  same folder: 0.3
  same label: 0.2
neighbor_coherence@10 = average over anchors of summed relation weights among top-10 neighbors divided by 10.
```

Bridge candidate quality score:

```text
bridge_candidate_quality_score = 1.0 if 5% to 35% of canonical decoded tracks are bridge candidates.
It decays linearly to 0 outside that range.
This is diagnostic only.
It must not enter default_composite_score, fallback_composite_score, or cluster_usability_score unless the user explicitly changes the objective.
```

Cluster guardrails:

```text
largest primary cluster should be <= 40% of canonical decoded tracks.
raw noise before reassignment should be <= 45% while N < 500, only for clusterers with native noise.
raw noise target tightens to <= 35% when N >= 500.
more than 30% of primary clusters with fewer than 5 tracks triggers penalty.
weak_primary rate is diagnostic and export-relevant only; if weak_primary rate > 20%, add review_warning = high_weak_primary_rate, but do not subtract from composite by default.
duplicate exports are hard failures.
```

Penalties:

```text
cluster_giant_penalty if largest primary cluster > 40%.
small_cluster_penalty if > 30% of primary clusters have fewer than 5 tracks.
no weak_primary penalty is applied by default because weak_primary is percentile-based and diagnostic; report weak_primary rate, raw primary margins, and primary_margin_percentile distribution.
duplicate_export_penalty as hard fail.
raw_noise_penalty only when raw_noise_rate is defined.
```

Done when:

1. metrics are computed for every competing config;
2. diagnostics are computed for diagnostic-only configs where possible;
3. component columns are present in the leaderboard;
4. penalties are transparent;
5. tests for metric formulas pass.

Decision rules:

1. If a component cannot be computed, mark unavailable and renormalize if it belongs to composite.
2. If a config produces duplicate exports, it cannot be winner.
3. If a clusterer has no native noise, skip raw-noise criterion but still report weak_primary, singleton count, primary margin distribution, and cluster-size guardrails.

### Task D5.3. Pick Regime 1 winner

Actions:

1. implement `scripts/dj_clustering/pick_winner.py`;
2. select winner from full leaderboard;
3. apply deterministic tie-breakers;
4. write `winner_selection.md`;
5. write machine-readable winner metadata.

Winner rule:

```text
best = config with highest default_composite_score.
If fewer than 50 validated 1001 positives exist in validation, best = config with highest fallback_composite_score.
```

Tie-breakers:

```text
1. higher manual_triplet_accuracy_validation
2. higher bootstrap_stability_score
3. lower raw noise rate when defined for both configurations
4. if raw_noise_rate is not_applicable for one or both configurations, skip raw-noise tie-breaker
5. lower weak_primary rate
6. simpler model, in this order: audio_only, audio_plus_bpm_key_light, audio_plus_metadata_light, audio_plus_bpm_key_metadata_light
```

Minimum success criteria:

```text
manual_triplet_accuracy_validation >= 0.65
cluster_usability_score >= 0.70
no duplicate exports
largest primary cluster <= 40%
if raw noise is defined: raw_noise_rate <= 45% for N < 500 or <= 35% for N >= 500
weak_primary rate is reported as diagnostic and does not determine pass/fail by default
```

Conservative noise exception:

```text
If raw_noise_rate > threshold but noise_policy = no_reassignment and manual_triplet_accuracy_validation >= 0.70,
the configuration may pass with flag conservative_noise_outlier.
This does not apply to forced reassignment policies.
```

Done when:

1. winner metadata exists;
2. winner rationale is written;
3. all tie-breakers are documented;
4. minimum success criteria status is recorded;
5. tracker records selected config ID.

Decision rules:

1. If no config passes minimum targets, continue to D7 fallback path after D6 review attempts.
2. If a config passes metrics but is diagnostic-only, it cannot be selected as winner.
3. Do not use held-out test accuracy to pick or reject the winner.

Cut D5 gate: full sweep and winner selection must be complete before overlapping cluster export.

## 15. Phase 6. Overlapping clustering and export

Goal: generate the actual folder, auxiliary cluster, and bridge manifests from the selected Regime 1 configuration.

### Task D6.1. Assign overlapping clusters

Actions:

1. implement `scripts/dj_clustering/assign_overlapping_clusters.py`;
2. implement or update `src/dj_clustering/clustering.py`;
3. load winner config;
4. compute primary membership scores;
5. compute secondary membership scores;
6. calibrate membership percentiles within the selected config;
7. assign primary clusters;
8. assign up to two auxiliary clusters;
9. compute bridge scores;
10. flag weak primary tracks;
11. write `cluster_membership.csv`.

Membership source policy:

```text
HDBSCAN with hdbscan package and prediction_data=True:
  use native soft membership if available.

HDBSCAN with sklearn or precomputed distances where native membership is unavailable:
  use medoid-distance fallback membership and record membership_source = medoid_fallback.

KMeans:
  use cosine similarity to cluster centroid.

Agglomerative:
  use cosine similarity to medoid, where medoid is the track with lowest mean distance inside the cluster.
```

Auxiliary cluster rule:

```text
Compute raw_m1 and raw_m2.
Compute ratio = raw_m2 / raw_m1 when raw_m1 > 0.
Compute percentile rank of raw_m2 within the config.
Compute percentile rank of ratio within the config.
Assign second cluster if:
  secondary_confidence_percentile >= 0.70
  AND ratio_percentile >= 0.70
  AND the cluster differs from primary.
Assign third cluster only if it also passes the same thresholds and max auxiliary cap is not exceeded.
```

Bridge score:

```text
bridge_score = secondary_confidence_percentile * secondary_ratio_percentile
```

This formula avoids raw-score comparability problems across clusterer families. Raw scores and raw ratios are stored only for diagnostics.

Weak primary rule:

```text
weak_primary if:
  native raw-noise status before reassignment
  OR primary_confidence_percentile < 0.20
  OR primary_margin_percentile < 0.20
```

Precedence rule:

```text
If weak_primary and bridge_candidate are both true, weak_primary takes precedence for export semantics.
The track is ambiguous, not a reliable bridge.
```

Done when:

1. `cluster_membership.csv` exists;
2. every canonical decoded track has one primary cluster unless explicitly marked uncertain;
3. zero to two auxiliary clusters are assigned per track;
4. bridge score is computed;
5. weak primary flag is computed;
6. membership source is recorded;
7. tests for auxiliary thresholds and bridge precedence pass.

Decision rules:

1. If a track has primary_score below threshold, keep it in manifest with `weak_primary = true`.
2. If clusterer creates singletons, export them to `_singletons/` manifest and report as over-clustering diagnostic.
3. If more than 20 percent of tracks are `weak_primary`, add `review_warning = high_weak_primary_rate` and ensure `cluster_report.md` includes raw primary margins, the primary margin percentile distribution, and representative uncertain examples.
4. This warning does not fail D6 by itself. D6 may fail only through manual sample review, duplicate export failures, missing required manifests, or explicit user rejection of the exported organization.

### Task D6.2. Export cluster manifests and M3U playlists

Actions:

1. implement `scripts/dj_clustering/export_cluster_manifests.py`;
2. implement or update `src/dj_clustering/export.py`;
3. create primary folder manifest;
4. create auxiliary cluster manifests;
5. create `bridge_tracks.csv`;
6. create `_uncertain/` manifest for weak primary tracks;
7. create `_singletons/` manifest for singleton clusters;
8. restore exact duplicate rows at export time according to canonical assignment;
9. write optional M3U playlists using absolute paths by default;
10. write `cluster_report.md`.

Export semantics:

```text
Primary folder manifest:
  includes canonical decoded tracks with primary assignment.

_uncertain manifest:
  includes weak_primary tracks.
  These may still have a primary assignment in the manifest, but are flagged for review.

_singletons manifest:
  includes singleton cluster tracks.

bridge_tracks.csv:
  includes reliable bridge candidates only.
  weak_primary tracks do not become reliable bridges.

Auxiliary M3U playlists:
  include tracks with auxiliary cluster assignments.

Exact duplicates:
  are restored at export time by copying the canonical assignment to duplicate rows.
```

Physical audio export default:

```text
Do not copy or duplicate audio files by default.
Export manifests and M3U playlists.
Physical folder copy mode requires explicit user approval in a later task.
```

Cluster naming policy:

```text
For each cluster, compute genre_vote = top-1 normalized genre among tracks in the cluster.
If genre_vote covers >= 60% of tracks and does not duplicate another cluster name in the same run, use it as a metadata-derived name.
Otherwise use Cluster_01, Cluster_02, etc.
```

Done when:

1. primary manifest exists;
2. auxiliary manifests exist;
3. bridge report exists;
4. uncertain and singleton manifests exist if needed;
5. exact duplicates are restored only at export time;
6. no audio files are duplicated by default;
7. M3U paths follow configured path policy;
8. duplicate export check passes.

Decision rules:

1. If duplicate exports are detected, fail the task and fix export logic.
2. If cluster names are non-distinctive, use neutral names.
3. If full file paths are private, keep M3U files outside Git and commit only sanitized summaries.

### Task D6.3. Manual sample review of exported clusters

Actions:

1. select 5 clusters for manual review;
2. write `cluster_review_sample.csv`;
3. write M3U files for sampled clusters if possible;
4. ask user to mark each sampled cluster as `usable`, `partially_usable`, or `not_usable`;
5. ingest review;
6. write `manual_cluster_review.md`.

Sample selection rule:

```text
Select 5 clusters:
  the largest non-uncertain primary cluster;
  the smallest non-singleton non-uncertain primary cluster;
  one median-size non-uncertain primary cluster;
  one cluster with high bridge_candidate_rate;
  one cluster with low bridge_candidate_rate.
```

Pass rule:

```text
usable = 1.0
partially_usable = 0.5
not_usable = 0.0
Pass if total score >= 4.0 out of 5.0.
```

Done when:

1. review sample exists;
2. user review is recorded or explicitly pending;
3. pass or fail status is recorded;
4. tracker records outcome.

Decision rules:

1. If user review is pending, mark task blocked rather than complete.
2. If review fails, proceed to D7 alternative winner path.
3. Do not use held-out test triplets to trigger alternative winner selection.

Cut D6 gate: overlapping cluster export and manual sample review must pass before final evaluation.

## 16. Phase 7. Final evaluation, fallback, and Regime 2 trigger

Goal: evaluate the selected system without test leakage, define fallback behavior, and only run Regime 2 when justified.

### Task D7.1. Final evaluation of locked winner

Actions:

1. lock final winner config ID;
2. evaluate held-out manual triplets exactly once;
3. evaluate held-out 1001 test split if it exists;
4. summarize validation vs test results;
5. summarize export review results;
6. write `reports/dj_clustering/final_evaluation.md`.

Held-out test rule:

```text
Held-out test is final-report-only.
It must not trigger T12-style alternative winner selection.
It must not be used for sweep, tie-breakers, or Regime 2 hyperparameter selection.
```

Done when:

1. final evaluation report exists;
2. held-out test metrics are reported only after winner lock;
3. limitations are stated;
4. tracker records final status.

Decision rules:

1. If validation passed but held-out test is weak, do not automatically switch winners.
2. Document the weakness and recommend more evidence or Regime 2 only in a new cycle.
3. If held-out test was not available, report that limitation.

### Task D7.2. Alternative winner path if validation or user review fails

Actions:

1. select second-best eligible config from leaderboard;
2. rerun D6.1 and D6.2 for that config;
3. rerun D6.3 manual sample review;
4. repeat for top-3 configs only;
5. write alternative winner report.

Fallback loop:

```text
D6 fail by validation metrics or user review -> D7.2 alternative winner.
Try up to top-3 eligible configs.
If one passes, lock it and proceed to D7.1.
If top-3 exhausted, proceed to D7.3 or D7.4.
```

Done when:

1. alternative candidates are tried or explicitly skipped;
2. reason for each failure is recorded;
3. selected alternative or exhaustion status is recorded.

Decision rules:

1. Maximum 3 winner attempts.
2. Do not use held-out test to choose among alternatives.
3. If all top-3 fail, collect more evidence or trigger Regime 2 if allowed.

### Task D7.3. Collect more evidence if Regime 1 evidence is insufficient

Actions:

1. generate 40 additional active triplets;
2. ingest user answers;
3. assign new labels to train or validation only;
4. do not move existing held-out test triplets;
5. rerun full sweep from D5.1;
6. write additional evidence report.

Done when:

1. additional triplets are collected or user declines;
2. split policy is respected;
3. tracker records new evidence count.

Decision rules:

1. If evidence is below Regime 2 threshold, prefer D7.3 over D7.4.
2. If user declines more labels, mark limitation and either accept best available config or stop.
3. Do not relabel held-out test examples.

### Task D7.4. Run Regime 2 projection head only if justified

Actions:

1. verify Regime 2 evidence threshold;
2. create `configs/dj_clustering/regime2_projection_head.yaml`;
3. implement projection head training only over frozen embeddings;
4. train using train split only;
5. select hyperparameters using validation split only;
6. avoid treating relative negatives as absolute negatives;
7. rerun D5, D6, and D7 with Regime 2 similarity space;
8. write Regime 2 report.

Regime 2 trigger:

```text
Run Regime 2 only if no Regime 1 configuration passes minimum targets after top-3 attempts AND evidence threshold is met.
```

Evidence threshold:

```text
Enable Regime 2 if:
  manual non-skip triplets >= 200
OR
  manual non-skip triplets >= 100 AND validated 1001 positives with confidence >= 0.7 are >= 1000.
```

Negative policy:

```text
Explicit manual relative negatives are preferred.
Unlabeled nearest-neighbor hard negatives may be used only as low-weight pseudo-negatives.
Do not assign them the same weight as manual negative evidence.
For each positive pair, hard pseudo-negatives may be sampled as k=5 unlabeled tracks with smallest cosine distance to the anchor under current embedding.
Mark them as pseudo_negative with low weight.
```

Done when:

1. Regime 2 either runs or is explicitly skipped because threshold is not met;
2. training uses train split only;
3. validation split selects hyperparameters;
4. held-out test remains final-report-only;
5. report states whether Regime 2 improved over Regime 1.

Decision rules:

1. If evidence threshold is not met, do not train projection head.
2. If projection head overfits validation or worsens cluster review, keep Regime 1 as selected result or mark project limited.
3. Full encoder fine-tuning remains out of scope.

Cut D7 gate: final evaluation or documented fallback decision must be complete before the branch is considered done.

## 17. Final branch definition of done

The DJ clustering branch is done when:

1. plan file exists in `docs/plans/dj_clustering_plan.md`;
2. progress trackers are current;
3. local inventory and identity logic exist;
4. decode failures are handled without blocking the whole pipeline;
5. exact duplicates are detected and excluded from clustering by default;
6. default frozen features exist or missing feature blockers are documented;
7. pairwise features and similarity profiles are implemented;
8. manual triplet question queue and ingestion are implemented;
9. 1001Tracklists evidence is either implemented or explicitly marked unavailable;
10. Regime 1 sweep runs with conditional grid and fixed baselines;
11. validation metrics and guardrails select a winner without test leakage;
12. overlapping clusters, auxiliary memberships, bridge scores, and weak primary flags are exported;
13. primary manifests, auxiliary manifests, bridge report, `_uncertain`, and `_singletons` outputs exist where applicable;
14. exact duplicate rows are restored only at export time;
15. no audio duplication occurs by default;
16. final evaluation report exists;
17. Regime 2 is either not needed, skipped with reason, or completed under the trigger rules;
18. all large and private artifacts remain outside Git;
19. branch is committed and pushed to GitHub.

## 18. Practical control rules

1. Execute one task at a time.
2. Do not skip tracker updates.
3. Do not commit private music files.
4. Do not commit downloaded 1001 or YouTube audio.
5. Do not commit embeddings, pairwise tables, model checkpoints, or large run folders.
6. Do not rely on current Traktor playlists as trusted supervision.
7. Do not modify or delete existing `playlists/V4_*` reference exports during D0-D7.
8. Do not commit new generated M3U files with private absolute paths unless the user explicitly approves a sanitized export.
9. Do not move current V4 to `legacy/` during D0-D7.
10. Do not refactor `src/v4/` in place unless a later user-approved cleanup task explicitly requests it.
11. Do not use V2 or V4 as ground truth.
12. Do not use 1001Tracklists as a clustering feature in Regime 1.
13. Do not use held-out test evidence for model selection.
14. Do not treat manual triplet non-chosen candidates as absolute negatives.
15. Do not allow KMeans to consume arbitrary precomputed pairwise similarity matrices.
16. Do not use ward linkage with precomputed agglomerative distances.
17. Do not force HDBSCAN noise reassignment without comparing explicit noise policies.
18. Do not claim success from the composite score alone without component metrics and export review.
19. Do not duplicate audio in export unless explicitly requested in a later task.
20. If generated code conflicts with this plan, follow this plan and record the conflict.
21. If this plan conflicts with `CLAUDE.md`, follow `CLAUDE.md` except for the explicitly approved separate namespace and frozen V4 policy; record the conflict in `docs/PROJECT_MAP.md`.
22. Stop at gates and report status.

## 19. First Claude Code task recommendation

The first implementation task should be D0.1 only.

Claude Code should not implement inventory, features, triplets, or clustering in the first task.

The first task should inspect repository state, confirm branch and remote status, check dirty files, identify where the plan and trackers should live, identify whether current V4 and legacy folders exist, and stop with a report.

