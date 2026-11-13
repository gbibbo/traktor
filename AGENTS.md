# AGENTS.md

Canonical instructions for coding agents in this repository.

## Start here

1. Read `docs/STATUS.md` first: current state, open items, and run recipes. Claude Code injects it at session start. Then read `README.md` for the V4 pipeline and project scope.
2. Read `docs/PROJECT_MAP.md`, `docs/V4_USAGE.md`, `docs/v4/TODO.md`, and `docs/DECISIONS.md` before substantial work. `docs/v4/JOBS_STATUS.md` is the legacy HPC/Slurm record from March 2026, not current state.
3. Inspect the actual Git repository, code, tests, configuration, generated artifacts, and dataset state before asserting that prior work or a milestone exists.
4. If you are helping a person reproduce the live suggestions inside Traktor on another computer,
   follow `PROMPT_LLM.md`. The setup guide is `docs/SUGERENCIAS_EN_VIVO.md`.
5. Treat `v4_implementation_plan.md` and `dj_music_clustering_deterministic_implementation_plan_v6.md` as design history and planning context. The current implementation under `src/v4/` and current documentation take precedence when they disagree with older plans.

## Project boundary

This is the independent TRAKTOR ML project for organizing a private Techno and Tech House collection into Traktor ready playlists through audio feature extraction, clustering, ordering, export, and visual inspection.

The current development environment is a local, CPU-first workstation. Local CPU development is the default: repository inspection, code editing, tests, Phase 0 ingestion, Phase 1 CPU extraction (`--essentia-only`, or `--percussion hpss --device cpu`), Phases 2-5, the Streamlit app, and Git operations all run locally without any cloud machine. Heavy work (GPU, more than about an hour of CPU, or more than about 10 GB of disk) goes to the gpuq GPU queue with its data on Gabriel's Google Drive; see the Compute section.

Lightning AI Studio, Surrey HPC, and Slurm are optional legacy/remote infrastructure. They must never be started, switched, or billed automatically. Do not assume any of them is available, do not submit Slurm jobs, and do not launch, switch, or resume a Lightning machine unless Gabriel explicitly asks for that infrastructure in the current conversation. The helper scripts under `tools/lightning/` and the jobs under `slurm/` are retained for provenance only and are inert by default.

The private audio collection, generated audio features, embeddings, model caches, and large artifacts must remain outside Git. Never commit source music, generated stems, `.npy`, `.parquet`, checkpoints, caches, credentials, `.env` files, API keys, or other secrets.

## Current dataset and environment

* Primary working dataset: `test_20`.
* Expected source audio count: 243 tracks.
* The full-library MVP runs on dataset `musica` (`Música/` at the repo root, ignored by Git); `test_20` remains the evaluation set with triplets and Gabriel's own folders. See `docs/STATUS.md`.
* Default audio location: `data/raw_audio/test_20/` (resolved relative to the repo root). `data/` is ignored by Git; on a local machine this path is normally a filesystem link (symlink or Windows directory junction) to the real music folder, never a committed copy. It must never be committed.
* Python environment: Python 3.11. On Windows the CPU pipeline (Phases 0, 2-5, tests, UI) runs on a local Python 3.11 interpreter or `.venv`. Essentia (Phase 1 CPU BPM/key) has no native-Windows build, so Phase 1 CPU extraction runs under WSL/Linux with Python 3.11.
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

Before saying that something is missing on a computer, or that it has to be copied or downloaded there, search that computer for it. Look in the repo, the other drives and the pendrive, and for copies under other names or folder layouts; `docs/STATUS.md` says where the copies live. Match by content (track UIDs, hashes, counts), not by path. "The code does not find it at path X" is a different claim from "it is not on this computer": say which one you checked and where you looked. Before proposing to copy or download anything, offer to point the code at the copy that already exists.

For audits, diagnoses, and comparisons between representations, parameters, or organizations, follow `.claude/skills/auditar/SKILL.md` (Claude Code loads it as the `auditar` skill; other agents read the file). A better triplet score is not a better organization: between representations, Gabriel decides by listening.

When a new conclusion kills an old one in the docs, edit the old line in place instead of deleting it: prefix it with `[REFUTADO <date> by <evidence>: <what is true>]` if it was wrong, or `[REEMPLAZADO <date> by <decision or change>: <what applies now>]` if it was right but no longer applies.

## Git discipline

Use Git as the durable checkpoint mechanism.

After a coherent verified unit of work is complete, validate it, update relevant durable documentation, commit it, and push the current branch when a valid upstream exists.

Rules:

* Commit coherent verified units, not every tiny edit.
* Push normal commits to the current branch without asking each time when a valid upstream exists.
* Do not force push, rewrite shared history, merge branches, delete remote branches, or open or merge pull requests unless Gabriel asks.
* Never commit datasets, music files, generated stems, embeddings, large artifacts, credentials, `.env` files, API keys, or secrets. The pre-commit check in `tools/git_hooks/` enforces this list (activate once per clone with `git config core.hooksPath tools/git_hooks`). Never bypass it with `--no-verify`; if it flags correct work, fix `tools/git_hooks/check_staged.py` and its tests.
* Gabriel's parallel work lives in `download_JIJIJI/`: include it in commits and pushes, never its audio files (`.gitignore` covers the audio extensions). The pre-commit check scans staged content for credentials; still look at what you stage.
* Do not add AI attribution, generated with trailers, or session links to commits or pull request text.
* If tests are failing or evidence is incomplete, record that state instead of disguising it as completed work.

Before stopping after substantive project work, report the final Git state with the full HEAD SHA, push status, and whether the working tree is clean. Do not create empty commits merely to manufacture a SHA.

### Several sessions on one working tree

Several agent sessions work at the same time in this one working tree. They share its files, its Git index and the branch `main`. Sessions coordinate among themselves; Gabriel is not asked to sort out commits or pushes.

There is one way to commit and push:

```text
python tools/repo/subir.py -m "<message>" <path> [<path> ...]
python tools/repo/subir.py --estado
```

The tool waits its turn on a lock, commits only the paths you name through its own index, runs the pre-commit check on that commit, pushes, and releases the lock. Whatever another session has staged or left unsaved in other files stays exactly as it was. A plain `git commit` is rejected by the pre-commit hook with a pointer to this tool; nothing is lost when that happens.

Rules:

* Name the paths you changed. The tool refuses the repository root.
* Do not run these in the shared tree, because they move or delete other sessions' unsaved work: `git stash`, `git switch` or `git checkout <branch>`, `git checkout -- <file>`, `git restore`, `git reset --hard`, `git clean`, `git pull`, `git rebase`, `git merge`, `git add -A`, `git commit -a`. Branch work goes in its own `git worktree`. Read-only commands (`status`, `diff`, `log`, `show`) are always fine.
* A file is committed whole. Shared documents (`docs/STATUS.md`, `docs/DECISIONS.md`, `docs/PROJECT_MAP.md`, `docs/LESSONS_LEARNED.md`, the plan and its inventory) are edited by adding complete lines in one edit and committed right away, in the same step. If the file carries finished lines from another session, they go up with yours: that is expected and loses nothing. Do not separate blocks by hand with `git apply --cached` or similar.
* If a shared document fails its check because of lines that are not yours (for example `STATUS.md` over its size limit), wait a minute and run the tool again. Do not edit or remove another session's lines.
* A code file has one owner session at a time. If `--estado` shows a file as modified and you did not modify it, another session is working on it: message that session (`SendMessage`) instead of editing it.
* Exit code 3 means the lock stayed busy: run the same command again. Exit code 4 means the remote has commits this clone lacks: stop and coordinate with the other sessions. Exit code 5 means the commit was rejected, and the output says why. `PUSH pendiente` means the commit is safe locally and goes out with the next run; report it in the checkpoint.
* If the tool itself fails, fix `tools/repo/subir.py` and its tests in `tests/test_subir.py`. Do not go around it.

## Compute: local CPU, gpuq GPUs, and the Drive

Gabriel's rule (`docs/DECISIONS.md` 2026-11-09, recommendation algorithm, point 7): light work runs on this workstation's CPU; anything heavy in processing or in disk runs on gpuq GPUs with its data on his work Google Drive. Setup, commands, and the end-to-end test: `docs/GPUQ.md`.

### 1. Light or heavy

* **Light, local CPU:** repository inspection, code editing, tests, Phase 0 ingestion, clustering, ordering, export, Streamlit, metadata, manifests, hashing, plots, documentation, Git, benches on stored embeddings, CLAP text, and extractions on a small folder of tracks.
* **Heavy, gpuq and Drive:** anything that needs a GPU, or would take more than about one hour of CPU on this workstation, or more than about 10 GB of new disk (the agent's thresholds; Gabriel may change them). Typical case: a model pass (Demucs, MERT, MAEST, CLAP, Essentia heads) over the whole library or over the DJ-set blocks.
* When in doubt, time a small sample on CPU and extrapolate. The measured number decides, not a guess.

### 2. How gpuq is wired here (verified 2026-11-09)

* The dispatcher runs in Docker (container `gpuq`) on Gabriel's always-on Raspberry Pi `asr-rp5`. It launches jobs on Colab (one free T4 at a time) and Kaggle (about 30 GPU hours a week, 12 hours per session), relaunches them when a machine is lost, and notifies Gabriel's phone.
* **Before sending work to Colab, read `docs/GPUQ.md`, section «Colab».** Since 2026-11-02 Colab reclaims every T4 VM that runs a real gpuq job 12-14 min after assignment (still so on 11-13), while a GPU-only test job lives past 30 min; it started on Colab's side, not in our jobs or gpuq's design. Each Colab attempt gives about 10 useful minutes, so only jobs that save every few minutes belong there and long work goes to `--platform kaggle`. That section says how to recognize the cut and when it is over (`docs/reports/colab_corte_13min_2026-11-13.md`).
* The queue lives in Gabriel's private repo `gbibbo/gpuq-personal` (queue on branch `jobs`, status and logs on branch `runs`). This workstation has a clone next to this repo, `../gpuq-personal` (override with `GPUQ_HOME`). Queue and follow jobs from this repo with `tools/gpuq/cola.py` (`add`, `status`, `logs <id>`, `cancel <id>`).
* A job checks out **this repository at a pushed commit** (`cola.py add` pins the current HEAD and refuses a commit that is not on GitHub), runs its command at the repo root, and reaches the Drive through rclone.
* The Drive is the rclone remote `erclm`, Gabriel's work Google Drive, with scope `drive.file`: rclone sees only the files this rclone client created, never the rest of his Drive. The same remote is configured on this workstation. TRAKTOR ML data lives under `erclm:traktor_ml/`; `src/v4/common/remote_storage.py` resolves that root on the VM and here, and `open_remote()` returns a gpuq `Remote` whose transfers are atomic and verified by size and MD5. `ERC-LM-2027/` and `gpuq-test/` on the same remote belong to other projects: do not write or delete there.
* This repository is **public** on GitHub. Code goes to Git; data, results, and anything private go only to the Drive.

### 3. Procedure for a heavy job

1. **Write it resumable** (gpuq's `docs/WRITING_JOBS.md`): everything that must survive goes to the remote through `open_remote()`; work in shards and skip the shards already on the remote; checkpoint long loops with `gpuq.storage.Checkpoints`; print progress with `flush=True`.
2. **Validate on CPU with a small sample**, same code path, with `TRAKTOR_REMOTE` pointing at a local folder. Measure the time per item.
3. **Upload the inputs** under `erclm:traktor_ml/<task>/` (rclone or `open_remote().put`). Prefer derived data (embeddings, features) over audio. Audio from the collection goes up only with Gabriel's OK for that set.
4. **Commit and push** the job code.
5. **Ask Gabriel before the first submission of each kind of job**, in the current conversation, stating: what needs the GPU, the CPU time per item measured in step 2, the GPU hours against the free quota, the Drive space it will take (`rclone about erclm:`), and the remote paths. Later jobs of the same kind (same script, other shards or parameters) need no new OK.
6. **Run a short job first** (minutes), check `cola.py logs <id>` and the outputs on the remote, then queue the full job. For jobs longer than about an hour, interrupt one on purpose and check that it resumes.
7. **Bring the results back** with `open_remote().get` or rclone, check shapes, row counts, track UID alignment, and finite values, and record the job ids and remote paths in the report or in `docs/STATUS.md`.

### 4. Hard limits

* **Credentials never pass through the agent.** Gabriel enters them himself: `gpuq setup` on the Pi, and `tools/gpuq/rclone_desde_raspberry.ps1` to copy the Drive remote to another workstation. Never ask for, read, print, or commit a token: no `rclone config show`, no reading `rclone.conf` or `~/gpuq-data/secrets`. The agent may run `docker exec gpuq gpuq doctor` on the Pi and rclone commands that list, copy, or move data.
* **Nothing public:** no public Kaggle datasets or notebooks, no shared Drive links, and Gabriel's e-mail address is never written in this repository.
* **Do not change the dispatcher** (its `gpuq.toml` in `gpuq-personal`, the Pi, the container) without Gabriel's OK: it serves other projects too.
* **Paid compute** (Lightning, Colab compute units, any cloud credits), Slurm, and Surrey HPC are never started, switched, resumed, or billed without Gabriel's explicit approval in the current conversation. Before any paid GPU use: a CPU-validated code path, what needs the GPU, the smallest GPU class, a cost bound, and his OK.
* If a GPU failure reproduces on CPU, debug on CPU. GPU time, free or paid, is not a debugger.

### 5. Long runs on this workstation

Before any local run that would keep the machine busy for more than a few minutes, first run the same command on a small folder of tracks (for extraction: `extract_representations.py --folder "<subfolder>"`, which shares the cache) and check the output: no failures, finite values, plausible results, time per track. Only then launch the full run, with `--low-priority` so Gabriel can keep using Rekordbox/Traktor on the same machine. If the full run would cross the heavy threshold, it goes to gpuq instead.

## Coding standards

For nontrivial changes, inspect existing code before creating new scripts or parallel implementations. Extend or refactor the existing V4 path when practical.

New or materially edited Python files should begin with a concise module docstring describing `PURPOSE` and `CHANGELOG`, matching the existing repository convention. Use `pathlib` for filesystem paths. Keep path resolution centralized and portable across local Windows, WSL/Linux, and any optional remote environment; do not hardcode machine-specific absolute paths in tracked code. Console output must stay portable: `src/v4/common/__init__.py` forces UTF-8 stdout/stderr so non-ASCII output does not crash on Windows cp1252 consoles.

Keep `docs/STATUS.md` current: edit it in place when the state, open items, or run recipes change. It holds the final state, not a chronicle (history lives in `git log`, `docs/DECISIONS.md`, and `docs/reports/`), and stays under 6 KB (`tests/test_status_doc.py`).

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

Anything asked of Gabriel has to fit a busy person (his rule, 2026-11-13). Say beforehand how long it takes, keep it within 15 minutes, and put one decision per screen. Whatever an agent can decide with a default or check on its own is not asked: decide it, tell him in one line, and let him change it. Listening tests use the short format of `tools/playlist_review/pagina_escucha.py`. Full rule: `docs/reports/escucha_corta_e12_2026-11-13.md`.