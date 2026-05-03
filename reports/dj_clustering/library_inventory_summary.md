# Library Inventory Summary

## Run metadata

- Scan date (UTC): 2026-05-03 01:43 UTC
- Config: `configs/dj_clustering/inventory.yaml`
- Library root status: active (path withheld — see ignored artifact CSV)
- Recursive: true
- mutagen available: yes
- Decode backends: soundfile, torchaudio
- Hash method: sha256_full_content

## File counts

| Metric | Count |
|---|---|
| Total files scanned | 492 |
| Audio files (pass whitelist) | 246 |
| Non-audio files excluded | 246 |
| Hidden files skipped | 0 |

## Decode status

| Status | Count | Rate |
|---|---|---|
| ok | 246 | 100.0% |
| ok_probe_only | 0 | 0.0% |
| ok_torchaudio | 0 | 0.0% |
| failed | 0 | 0.0% |

## Identity

| Metric | Count |
|---|---|
| Unique audio hashes | 246 |
| Exact duplicate groups | 0 |
| Total duplicate files | 0 |
| Canonical tracks | 246 |
| Canonical decoded (N_canonical_decoded) | 246 |

## Extension distribution

| Extension | Count |
|---|---|
| mp3 | 245 |
| wav | 1 |

## Metadata quality

| Flag | Count |
|---|---|
| full_tags | 242 |
| partial_tags | 3 |
| no_tags | 1 |

## Metadata coverage (canonical decoded tracks only)

| Field | Non-null | Coverage |
|---|---|---|
| artist | 245 | 99.6% |
| title | 245 | 99.6% |
| album | 242 | 98.4% |
| genre | 244 | 99.2% |
| label | 240 | 97.6% |
| year | 230 | 93.5% |
| bpm | 245 | 99.6% |
| key | 0 | 0.0% |

## Folder hint distribution (top 10, canonical tracks)

Folder hint labels are anonymised. See ignored artifact CSV for actual names.

| Label | Count |
|---|---|
| folder_hint_001 | 243 |
| folder_hint_002 | 3 |

## D1.2 gate

| Metric | Value |
|---|---|
| Decode failure rate | 0.0% |
| Threshold | 20.0% |
| Gate | PASS |
| N_canonical_decoded | 246 |

_Artifact: `artifacts/dj_clustering/inventory/library_inventory.csv` (git-ignored)_
