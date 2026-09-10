# AGENTS.md

Canonical instructions for coding agents in this repository.

## Start here

1. Read `README.md` for the current V4 pipeline and project scope.
2. Read `docs/PROJECT_MAP.md`, `docs/V4_USAGE.md`, `docs/v4/TODO.md`, and `docs/v4/JOBS_STATUS.md` before substantial work.
3. Inspect the actual Git repository, code, tests, configuration, generated artifacts, and dataset state before asserting that prior work or a milestone exists.
4. Treat `v4_implementation_plan.md` and `dj_music_clustering_deterministic_implementation_plan_v6.md` as design history and planning context. The current implementation under `src/v4/` and current documentation take precedence when they disagree with older plans.

## Project boundary

This is the independent TRAKTOR ML project for organizing a private Techno and Tech House collection into Traktor ready playlists through audio feature extraction, clustering, ordering, export, and visual inspection.

The current development environment is Lightning AI Studio. The repository lives at `/teamspace/studios/this_studio/traktor`.

Historical Surrey HPC and Slurm files remain in the repository for provenance and reference. Do not assume Surrey infrastructure is available. Do not submit Slurm jobs unless Gabriel explicitly asks to use that infrastructure.

The private audio collection, generated audio features, embeddings, model caches, and large artifacts must remain outside Git. Never commit source music, generated stems, `.npy`, `.parquet`, checkpoints, caches, credentials, `.env` files, API keys, or other secrets.

## Current dataset and environment

* Primary working dataset: `test_20`.
* Expected source audio count: 243 tracks.
* Default audio location in Lightning: `data/raw_audio/test_20/`.
* Python environment: repository local `.venv`, Python 3.11.
* Dependency source: `requirements_v4.txt`.
* Main config: `config/v4.yaml`.

Before running project code, verify that the intended interpreter is active and that required imports succeed. Do not silently install or upgrade unrelated packages to solve an environment problem. Prefer the smallest change that restores the declared environment.

## Pipeline invariants

The V4 pipeline is split into explicit phases and agents must preserve those boundaries unless a documented redesign is requested.

* Phase 0 scans and validates the collection and builds the catalog.
* Phase 1 extracts Essentia BPM and key features, Demucs stems, and MERT embeddings.
* Phase 1 merge establishes the canonical successful track set and row alignment.
* Phase 2 performs PCA, HDBSCAN clustering, and UMAP projection.
* Phase 3 assigns readable cluster names.
* Phase 4 orders tracks using embedding, BPM, and key compatibility.
* Phase 5 exports Traktor ready M3U playlists.
* The Streamlit app is for inspection and re export, not a replacement for the pipeline artifacts.

`track_uids.json` after successful feature extraction is the canonical row alignment source for downstream embeddings, metadata, clustering outputs, ordering, and export. Never infer alignment from filesystem order.

## Evidence and verification

Verify before asserting. Documentation is context, not proof. Re run the narrowest relevant command or inspect the underlying artifact when a claim matters.

Before declaring a coding unit complete, run the narrowest tests that exercise the changed path. Before declaring a larger milestone complete, run the relevant V4 validation tests and confirm that generated artifacts have the expected shapes, row counts, and finite values.

When a result is surprising, check path resolution, dataset cardinality, track UID alignment, failed audio files, model versions, and config values before changing the algorithm.

## Git discipline

Use Git as the durable checkpoint mechanism.

After a coherent verified unit of work is complete, validate it, update relevant durable documentation, commit it, and push the current branch when a valid upstream exists.

Rules:

* Commit coherent verified units, not every tiny edit.
* Push normal commits to the current branch without asking each time when a valid upstream exists.
* Do not force push, rewrite shared history, merge branches, delete remote branches, or open or merge pull requests unless Gabriel asks.
* Never commit datasets, music files, generated stems, embeddings, large artifacts, credentials, `.env` files, API keys, or secrets.
* Do not add AI attribution, generated with trailers, or session links to commits or pull request text.
* If tests are failing or evidence is incomplete, record that state instead of disguising it as completed work.

Before stopping after substantive project work, report the final Git state with the full HEAD SHA, push status, and whether the working tree is clean. Do not create empty commits merely to manufacture a SHA.

## Compute resource discipline

Default to CPU. Use CPU for repository inspection, code editing, tests, ingestion, clustering, ordering, export, Streamlit work, metadata processing, manifests, hashing, plotting, documentation, and Git operations whenever reasonable.

GPU is an exception. For the current V4 pipeline, Phase 1 feature extraction is the main GPU justified stage because it runs Demucs and MERT over the collection. Do not keep a GPU active for downstream CPU work.

Before any paid GPU use:

1. Confirm that the code path is CPU validated as far as reasonably possible.
2. State what specifically requires GPU.
3. Select the smallest compatible GPU class.
4. Estimate or bound the expected cost when paid compute is involved.
5. Obtain Gabriel's explicit approval before launching a paid GPU run.

If a GPU failure can be reproduced on CPU, return to CPU for debugging. Do not use paid GPU time as a debugger.

## Coding standards

For nontrivial changes, inspect existing code before creating new scripts or parallel implementations. Extend or refactor the existing V4 path when practical.

New or materially edited Python files should begin with a concise module docstring describing `PURPOSE` and `CHANGELOG`, matching the existing repository convention. Use `pathlib` for filesystem paths. Keep path resolution centralized and compatible with Lightning and local Windows export requirements.

Update `docs/PROJECT_MAP.md` when adding or removing important files or changing repository architecture. Add durable operational lessons to `docs/LESSONS_LEARNED.md` only when they are genuinely reusable and not duplicates of existing entries.

## Communication

Keep status concise and distinguish completed and verified work from planned work.

Every substantive final response should end with a checkpoint block in this form:

```text
## CHECKPOINT
COMMIT: <full SHA or NO NEW COMMIT>
PUSH: <status>
TREE: <clean or dirty>
```

For genuinely read only work, report `NO NEW COMMIT` and the current HEAD instead of creating a commit.