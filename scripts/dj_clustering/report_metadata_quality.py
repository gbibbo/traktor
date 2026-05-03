"""
PURPOSE: Generate a metadata quality report for the DJ clustering inventory.
  Reads the git-ignored inventory CSV, computes field coverage, class balance,
  and metadata sanity score validity, and writes a privacy-safe committed report
  to reports/dj_clustering/metadata_quality_report.md.

CHANGELOG:
  D1.3 (2026-05-03): Initial implementation.
"""

import argparse
import sys
from pathlib import Path

SANITY_THRESHOLD = 85.0  # percent: dominant class must not exceed this
BPM_VALID_COVERAGE = 90.0  # percent: minimum BPM coverage to treat as numeric feature

PRIVATE_PATTERNS = ["/mnt/fast", "nobackup", "scratch4weeks", "raw_audio"]


def _pct(count: int, total: int) -> float:
    return count / total * 100 if total > 0 else 0.0


def _miss(series, total: int) -> tuple[int, float]:
    n = int(series.isna().sum())
    return n, _pct(n, total)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate metadata quality report from inventory CSV."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("artifacts/dj_clustering/inventory/library_inventory.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dj_clustering/metadata_quality_report.md"),
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas not available; run inside traktor_ml conda environment")
        return 1

    if not args.inventory.exists():
        print(f"[ERROR] Inventory CSV not found: {args.inventory}")
        return 1

    print("[INFO] Reading inventory CSV ...")
    df = pd.read_csv(args.inventory)

    # Canonical decoded subset
    canonical = df[df["is_canonical"] & (df["decode_status"] != "failed")].copy()
    n_total = len(df)
    n_canonical = int(df["is_canonical"].sum())
    n_canonical_decoded = len(canonical)
    n_decode_failed = int((df["decode_status"] == "failed").sum())
    decode_failure_rate = _pct(n_decode_failed, n_total)
    n_exact_duplicates = int(df["is_exact_duplicate"].sum()) if "is_exact_duplicate" in df.columns else 0

    print(f"[INFO] Total rows: {n_total}")
    print(f"[INFO] Canonical decoded: {n_canonical_decoded}")
    print(f"[INFO] Decode failure rate: {decode_failure_rate:.1f}%")
    print(f"[INFO] Exact duplicates: {n_exact_duplicates}")

    # --- Decode status distribution ---
    decode_counts = df["decode_status"].value_counts(dropna=False).to_dict()

    # --- metadata_quality_flags distribution ---
    flag_counts = canonical["metadata_quality_flags"].value_counts(dropna=False).to_dict()

    # --- Metadata coverage ---
    coverage_fields = ["artist", "title", "album", "genre", "label", "year", "bpm", "key"]
    coverage: dict[str, tuple[int, float]] = {}
    for field in coverage_fields:
        if field in canonical.columns:
            non_null = int(canonical[field].notna().sum())
            coverage[field] = (non_null, _pct(non_null, n_canonical_decoded))
        else:
            coverage[field] = (0, 0.0)

    # --- Class balance for categorical fields (no class names printed or written) ---
    categorical_fields = ["genre", "artist", "label", "folder_hint"]
    sanity_results: dict[str, dict] = {}
    for field in categorical_fields:
        if field not in canonical.columns:
            sanity_results[field] = {
                "non_null": 0, "n_classes": 0, "dominant_count": 0,
                "dominant_pct": 0.0, "valid_for_sanity": False, "reason": "field absent",
            }
            continue
        series = canonical[field].dropna()
        non_null = len(series)
        if non_null == 0:
            sanity_results[field] = {
                "non_null": 0, "n_classes": 0, "dominant_count": 0,
                "dominant_pct": 0.0, "valid_for_sanity": False, "reason": "0 non-null values",
            }
            continue
        vc = series.value_counts()
        n_classes = len(vc)
        dominant_count = int(vc.iloc[0])
        dominant_pct = _pct(dominant_count, n_canonical_decoded)
        valid = n_classes >= 2 and dominant_pct <= SANITY_THRESHOLD
        if n_classes < 2:
            reason = "fewer than 2 classes"
        elif dominant_pct > SANITY_THRESHOLD:
            reason = f"dominant class covers {dominant_pct:.1f}% > {SANITY_THRESHOLD:.0f}% threshold"
        else:
            reason = "OK"
        sanity_results[field] = {
            "non_null": non_null, "n_classes": n_classes,
            "dominant_count": dominant_count, "dominant_pct": dominant_pct,
            "valid_for_sanity": valid, "reason": reason,
        }
        print(
            f"[INFO] {field}: {n_classes} classes, dominant {dominant_pct:.1f}%"
            f", valid_for_sanity={valid}"
        )

    # --- BPM distribution (continuous) ---
    bpm_series = canonical["bpm"].dropna() if "bpm" in canonical.columns else pd.Series(dtype=float)
    bpm_non_null = len(bpm_series)
    bpm_coverage_pct = _pct(bpm_non_null, n_canonical_decoded)
    bpm_valid = bpm_coverage_pct >= BPM_VALID_COVERAGE
    bpm_stats: dict[str, float] = {}
    if bpm_non_null > 0:
        bpm_stats = {
            "min": float(bpm_series.min()),
            "max": float(bpm_series.max()),
            "mean": float(bpm_series.mean()),
            "std": float(bpm_series.std()),
        }
    print(f"[INFO] bpm: coverage {bpm_coverage_pct:.1f}%, valid_as_numeric={bpm_valid}")

    # --- Key coverage ---
    key_non_null = int(canonical["key"].notna().sum()) if "key" in canonical.columns else 0
    key_coverage_pct = _pct(key_non_null, n_canonical_decoded)
    key_absent = key_non_null == 0
    print(f"[INFO] key: coverage {key_coverage_pct:.1f}%, absent={key_absent}")

    # --- Missingness profile ---
    def _col(name: str):
        return canonical[name] if name in canonical.columns else pd.Series([None] * n_canonical_decoded)

    artist_miss, artist_miss_pct = _miss(_col("artist"), n_canonical_decoded)
    title_miss, title_miss_pct = _miss(_col("title"), n_canonical_decoded)
    both_miss = int((_col("artist").isna() & _col("title").isna()).sum())
    both_miss_pct = _pct(both_miss, n_canonical_decoded)
    genre_miss, genre_miss_pct = _miss(_col("genre"), n_canonical_decoded)
    bpm_miss, bpm_miss_pct = _miss(_col("bpm"), n_canonical_decoded)
    key_miss = n_canonical_decoded - key_non_null
    key_miss_pct = _pct(key_miss, n_canonical_decoded)

    # --- Sanity score summary ---
    valid_sanity_fields = [f for f, r in sanity_results.items() if r["valid_for_sanity"]]
    invalid_sanity_fields = [f for f, r in sanity_results.items() if not r["valid_for_sanity"]]
    sanity_unavailable = len(valid_sanity_fields) == 0
    print(f"[INFO] Valid sanity field count: {len(valid_sanity_fields)}")
    print(f"[INFO] metadata_sanity_score_unavailable: {sanity_unavailable}")

    # --- Gate decision ---
    warnings: list[str] = []
    blocked_reasons: list[str] = []

    if n_canonical_decoded < 100:
        blocked_reasons.append(f"N_canonical_decoded={n_canonical_decoded} < 100")
    if decode_failure_rate >= 20.0:
        blocked_reasons.append(f"decode failure rate {decode_failure_rate:.1f}% ≥ 20%")

    if key_absent:
        warnings.append("key coverage is 0% — key excluded from feature candidates")
    if not sanity_results.get("folder_hint", {}).get("valid_for_sanity", True):
        warnings.append(
            "folder_hint invalid for sanity scoring: "
            + sanity_results["folder_hint"].get("reason", "")
        )
    if sanity_unavailable:
        warnings.append(
            "metadata_sanity_score_unavailable=true — "
            "no valid categorical sanity field; renormalize composite weights at D5"
        )

    if blocked_reasons:
        gate = "BLOCKED"
    elif warnings:
        gate = "PASS+WARN"
    else:
        gate = "PASS"

    print(f"[INFO] Gate: {gate}")
    for w in warnings:
        print(f"[WARN] {w}")
    for b in blocked_reasons:
        print(f"[ERROR] GATE BLOCKED: {b}")

    # --- Suitability assessment (aggregate language only) ---
    assessment_parts = [
        f"The inventory contains {n_canonical_decoded} canonical decoded tracks "
        f"(small-N regime: 100 ≤ N < 500). "
        f"Decode failure rate is {decode_failure_rate:.1f}% and exact duplicate count is {n_exact_duplicates}.",
    ]
    if bpm_valid:
        assessment_parts.append(
            f"BPM coverage is {bpm_coverage_pct:.1f}%; BPM is available as a numeric feature "
            "for downstream grouping and sweep configurations."
        )
    else:
        assessment_parts.append(
            f"BPM coverage is {bpm_coverage_pct:.1f}%; below the 90% threshold, "
            "use with caution as a numeric feature."
        )
    if key_absent:
        assessment_parts.append(
            "Key is absent from all tracks (0% coverage) and cannot be used as a feature "
            "or sanity signal in any downstream task. This is a data gap, not a blocker."
        )
    if valid_sanity_fields:
        assessment_parts.append(
            f"{len(valid_sanity_fields)} categorical field(s) pass the metadata sanity validity "
            "rule (≥2 classes, dominant class ≤85%) and may be used in metadata_sanity_score at D5."
        )
    if sanity_unavailable:
        assessment_parts.append(
            "No categorical field passes the sanity validity rule. "
            "metadata_sanity_score_unavailable must be set to true; "
            "composite weights must be renormalized at D5 to exclude the sanity component."
        )
    if not blocked_reasons:
        assessment_parts.append(
            "The dataset is sufficient to proceed to D1.4 and D2 feature extraction "
            "without manual metadata repair. Categorical metadata fields with class diversity "
            "may serve as auxiliary grouping signals but should not be treated as primary "
            "cluster labels, particularly given the absence of key data."
        )
    assessment = " ".join(assessment_parts)

    # --- Build report lines ---
    ok_count = decode_counts.get("ok", 0)
    decode_line = (
        f"ok {ok_count} ({_pct(ok_count, n_total):.1f}%)"
        + (f", failed {n_decode_failed} ({decode_failure_rate:.1f}%)" if n_decode_failed > 0 else "")
    )

    lines: list[str] = [
        "# Metadata Quality Report",
        "",
        "## Inventory basis",
        "",
        "- Input artifact: `artifacts/dj_clustering/inventory/library_inventory.csv`",
        f"- Total rows: {n_total}",
        f"- Canonical rows: {n_canonical}",
        f"- Canonical decoded: {n_canonical_decoded}",
        f"- Decode status: {decode_line}",
        f"- Exact duplicate groups: {n_exact_duplicates}",
        "",
        "## Metadata quality flags",
        "",
        "| Flag | Count |",
        "|---|---|",
    ]
    for flag in ["full_tags", "partial_tags", "no_tags"]:
        if flag in flag_counts:
            lines.append(f"| {flag} | {flag_counts[flag]} |")
    for flag, count in flag_counts.items():
        if flag not in ("full_tags", "partial_tags", "no_tags"):
            lines.append(f"| {flag} | {count} |")

    lines += [
        "",
        "## Metadata coverage (canonical decoded tracks)",
        "",
        "| Field | Non-null | Coverage |",
        "|---|---|---|",
    ]
    for field, (non_null, pct) in coverage.items():
        lines.append(f"| {field} | {non_null} | {pct:.1f}% |")

    lines += [
        "",
        "## Class balance and metadata sanity score validity",
        "",
        "Applies the plan rule: a field is valid for `metadata_sanity_score` only if it has",
        "≥ 2 classes and no single class covers > 85% of canonical decoded tracks.",
        "Class names are not reported — only aggregate statistics.",
        "",
        "| Field | Non-null | Classes | Dominant count | Dominant % | Valid for sanity | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for field, r in sanity_results.items():
        valid_str = "yes" if r["valid_for_sanity"] else "no"
        lines.append(
            f"| {field} | {r['non_null']} | {r['n_classes']} | {r['dominant_count']}"
            f" | {r['dominant_pct']:.1f}% | {valid_str} | {r['reason']} |"
        )

    lines += [
        "",
        "## BPM distribution (canonical decoded tracks)",
        "",
        f"- Non-null: {bpm_non_null} / {n_canonical_decoded} ({bpm_coverage_pct:.1f}%)",
        f"- Valid as numeric feature (coverage ≥ {BPM_VALID_COVERAGE:.0f}%): {'yes' if bpm_valid else 'no'}",
    ]
    if bpm_stats:
        lines += [
            "",
            "| Stat | Value |",
            "|---|---|",
            f"| Min | {bpm_stats['min']:.1f} |",
            f"| Max | {bpm_stats['max']:.1f} |",
            f"| Mean | {bpm_stats['mean']:.1f} |",
            f"| Std | {bpm_stats['std']:.1f} |",
        ]

    lines += [
        "",
        "## Key coverage",
        "",
        f"- Non-null: {key_non_null} / {n_canonical_decoded} ({key_coverage_pct:.1f}%)",
        "- Assessment: key is absent from all tracks; excluded from feature candidates.",
        "",
        "## Missingness profile (canonical decoded tracks)",
        "",
        "| Condition | Count | % |",
        "|---|---|---|",
        f"| Missing artist | {artist_miss} | {artist_miss_pct:.1f}% |",
        f"| Missing title | {title_miss} | {title_miss_pct:.1f}% |",
        f"| Missing artist AND title | {both_miss} | {both_miss_pct:.1f}% |",
        f"| Missing genre | {genre_miss} | {genre_miss_pct:.1f}% |",
        f"| Missing BPM | {bpm_miss} | {bpm_miss_pct:.1f}% |",
        f"| Missing key | {key_miss} | {key_miss_pct:.1f}% |",
        "",
        "## Metadata sanity score validity summary",
        "",
        f"- Valid fields: {', '.join(valid_sanity_fields) if valid_sanity_fields else 'none'}",
        f"- Invalid fields: {', '.join(invalid_sanity_fields) if invalid_sanity_fields else 'none'}",
        f"- `metadata_sanity_score_unavailable`: {'true' if sanity_unavailable else 'false'}",
    ]
    if sanity_unavailable:
        lines.append(
            "- Action: renormalize composite weights at D5 to exclude sanity score component."
        )

    lines += [
        "",
        "## Suitability assessment",
        "",
        assessment,
        "",
        "## Gate decision",
        "",
        "| Criterion | Value | Result |",
        "|---|---|---|",
        f"| N_canonical_decoded ≥ 100 | {n_canonical_decoded}"
        f" | {'PASS' if n_canonical_decoded >= 100 else 'BLOCKED'} |",
        f"| Decode failure rate < 20% | {decode_failure_rate:.1f}%"
        f" | {'PASS' if decode_failure_rate < 20.0 else 'BLOCKED'} |",
        f"| Exact duplicates | {n_exact_duplicates} | PASS |",
        f"| Key coverage | {key_coverage_pct:.1f}%"
        f" | {'WARN (absent)' if key_absent else 'PASS'} |",
        f"| Sanity fields available | {len(valid_sanity_fields)} valid"
        f" | {'WARN (none)' if sanity_unavailable else 'PASS'} |",
        "",
    ]
    if blocked_reasons:
        lines.append("**Overall: BLOCKED**")
        for b in blocked_reasons:
            lines.append(f"- {b}")
    elif warnings:
        lines.append("**Overall: PASS+WARN**")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("**Overall: PASS**")

    lines += [
        "",
        "_Artifact basis: `artifacts/dj_clustering/inventory/library_inventory.csv` (git-ignored)_",
        "",
    ]

    report_text = "\n".join(lines)

    # --- Safety: block if any private pattern slipped through ---
    for pat in PRIVATE_PATTERNS:
        if pat in report_text:
            print(f"[ERROR] Private pattern '{pat}' found in report — aborting write.")
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text)
    print(f"[INFO] Report written to: {args.output}")
    print(f"[INFO] Gate decision: {gate}")

    return 1 if gate == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
