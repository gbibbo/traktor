# DJ Clustering — Restore Guide (post Surrey exit)

This guide restores the DJ clustering project from the cluster-exit checkpoint
onto a new host. It covers a local CPU host, a paid GPU VM, and another HPC.

References: `dj_clustering_cluster_exit_plan.md`,
`dj_clustering_artifact_manifest.yaml`.

## 0. What you need

1. The GitHub repository (code, configs, tests, reports, trackers, these docs).
2. The portable archives from the checkpoint:
   - `dj_clustering_critical_20260528.tar.zst` (required)
   - `dj_clustering_optional_20260528.tar.zst` (optional)
   - `SHA256SUMS.txt`
3. The raw audio bytes, copied out-of-band by you to your chosen destination,
   plus the ignored per-file checksum manifest
   `raw_audio_checksums_20260528.txt` if you want to verify them.

The archives and the raw-audio manifest are **not** in Git. Carry them on your
own storage (external SSD or personal cloud). The committed manifest records
their sha256 so you can confirm you have the right bytes.

## 1. Clone the repository

```bash
git clone git@github.com:gbibbo/traktor.git
cd traktor
git checkout dj-clustering-surrey-exit-2026-05-31   # or feature/dj-clustering-v1
```

## 2. Verify and restore the archives

```bash
# Place the archives somewhere, e.g. ./_restore/
cd _restore
sha256sum -c SHA256SUMS.txt        # expect: OK for each archive

# Extract at the repo root (recreates the ignored artifacts/ and runs/ trees):
tar --zstd -xf dj_clustering_critical_20260528.tar.zst -C /path/to/traktor
tar --zstd -xf dj_clustering_optional_20260528.tar.zst -C /path/to/traktor   # optional
```

After extraction the resume payload lives under
`artifacts/dj_clustering/` and `runs/dj_clustering/` exactly as on the cluster.

## 3. Restore raw audio (optional, only if re-extracting features)

Copy your out-of-band audio backup to a chosen location, then point the
inventory config at it. To verify integrity against the checkpoint:

```bash
cd /your/restored/audio/root
find . -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/check.txt
# Compare /tmp/check.txt against raw_audio_checksums_20260528.txt
diff <(sort /tmp/check.txt) <(sort raw_audio_checksums_20260528.txt) && echo MATCH
```

The inventory content hashes also let you re-link restored audio to existing
inventory rows even if paths changed.

## 4. Environment

Create a Python environment with the project dependencies (the same packages
used on the cluster: numpy, pandas, pyarrow, scikit-learn, hdbscan, umap-learn,
soundfile, tqdm; MERT extraction additionally needs torch + transformers).

```bash
conda create -n traktor_ml python=3.11
conda activate traktor_ml
pip install -r requirements_dj_clustering.txt   # if present; otherwise install the packages above
```

## 5. Functional check

```bash
conda run -n traktor_ml pytest tests/dj_clustering/ -q
```

A green suite confirms the restored code is consistent.

## 6. Continuation paths

### 6a. CPU-only (primary path)

Everything except feature re-extraction and retraining runs on CPU, because
the embeddings and the pairwise table are already restored:

- **D4.3 answer ingestion** — fill the active answer template, then:
  ```bash
  python scripts/dj_clustering/ingest_manual_triplets.py ingest \
      --merge-with artifacts/dj_clustering/triplets/manual_triplets.csv ...
  ```
- **Clustering sweeps** — `python scripts/dj_clustering/run_similarity_sweep.py ...`
- **Report updates** — the report generators read existing artifacts.

### 6b. Paid GPU VM

Use a generic Linux GPU VM. Steps 1–5 are identical. Reach for the GPU **only**
to re-extract MERT features (`scripts/dj_clustering/extract_features.py`) or to
train a Regime-2 projection head. Otherwise stay on the CPU path.

### 6c. Another HPC

Same as the GPU VM. If the cluster uses containers, the cluster-era backend was
a PyTorch CUDA image; any equivalent torch + transformers environment works for
re-extraction. No cluster-specific path is required for the CPU continuation.

## 7. Scientific status on resume

D4.3 stays blocked until the human active triplet answers are provided — an
offline human task, not a cluster dependency. The first-sweep results are
exploratory; no winner is selected and no final claim is made until D4.3/D5
evidence is restored and run.
