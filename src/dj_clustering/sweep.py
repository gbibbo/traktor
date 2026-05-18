"""
PURPOSE: Regime 1 similarity sweep planning for DJ clustering (Task D4.1).
         Provides deterministic grid expansion over the committed vector grid,
         conditional clusterer-branch construction, invalid-combination
         validation, seed-13 capped sampling, fixed-baseline injection, and
         diagnostic-only reference rows for V2/V4. No clustering or metric
         computation happens here; this module only builds the run plan.

         Regime 1 has no similarity_profile sweep axis: every config clusters
         directly on frozen MERT embeddings. The audio_only and
         audio_plus_metadata_light pair profiles are diagnostic references only.

CHANGELOG:
  D4.1a - Initial implementation (scaffold; sweep not executed).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import yaml

# Clusterers that have no native noise concept.
NOISE_NOT_APPLICABLE = "not_applicable"

# Reference systems represented as diagnostic-only rows. They are never
# re-run; V2 has no reusable embeddings and V4 artifacts are unavailable.
DIAGNOSTIC_REFERENCES = ("V2", "V4")


def load_sweep_config(path: Path) -> Dict:
    """Load and lightly validate the committed Regime 1 sweep config YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = [
        "embedding_sources",
        "mert_aggregations",
        "normalization",
        "dim_reduction",
        "clusterers",
        "hdbscan",
        "kmeans",
        "agglomerative",
        "sampling",
        "fixed_baselines",
        "baseline_definition",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"sweep config missing required keys: {missing}")
    return config


def config_id(cfg: Mapping) -> str:
    """Return a deterministic, filesystem-safe identifier for a run spec."""
    kind = cfg.get("kind", "grid")
    if kind == "baseline":
        return f"baseline.{cfg['baseline_name']}"
    if kind == "diagnostic":
        return f"diagnostic.{cfg['reference_name']}"
    parts = [
        cfg["embedding_source"],
        cfg["mert_aggregation"],
        f"norm_{cfg['normalization']}",
        cfg["dim_reduction"],
        cfg["clusterer"],
    ]
    if cfg["clusterer"] == "hdbscan":
        parts += [
            f"mcs{cfg['hdbscan_min_cluster_size']}",
            f"ms{cfg['hdbscan_min_samples']}",
            cfg["noise_policy"],
        ]
    elif cfg["clusterer"] == "kmeans":
        parts += [f"k{cfg['kmeans_k']}"]
    elif cfg["clusterer"] == "agglomerative":
        parts += [f"k{cfg['agglomerative_k']}", cfg["agglomerative_linkage"]]
    return ".".join(str(p) for p in parts)


def _base_combos(config: Mapping) -> List[Dict]:
    """Cross product of the shared axes, before the clusterer branch."""
    combos: List[Dict] = []
    for source in config["embedding_sources"]:
        for aggregation in config["mert_aggregations"]:
            for norm in config["normalization"]:
                for dim in config["dim_reduction"]:
                    combos.append(
                        {
                            "embedding_source": source,
                            "mert_aggregation": aggregation,
                            "normalization": norm,
                            "dim_reduction": dim,
                        }
                    )
    return combos


def expand_grid(config: Mapping) -> List[Dict]:
    """Deterministically expand the committed grid into clusterer run specs.

    Returns one dict per valid non-baseline config. Generation follows the
    conditional clusterer branches, so only valid combinations are produced;
    `is_valid_config` is still applied as a defensive guard.
    """
    hdb = config["hdbscan"]
    km = config["kmeans"]
    agg = config["agglomerative"]
    specs: List[Dict] = []

    for base in _base_combos(config):
        for clusterer in config["clusterers"]:
            if clusterer == "hdbscan":
                for mcs in hdb["min_cluster_size"]:
                    for ms in hdb["min_samples"]:
                        for policy in hdb["noise_policy"]:
                            specs.append(
                                {
                                    **base,
                                    "kind": "grid",
                                    "clusterer": "hdbscan",
                                    "hdbscan_min_cluster_size": mcs,
                                    "hdbscan_min_samples": ms,
                                    "noise_policy": policy,
                                    "kmeans_k": None,
                                    "agglomerative_k": None,
                                    "agglomerative_linkage": None,
                                }
                            )
            elif clusterer == "kmeans":
                for k in km["k"]:
                    specs.append(
                        {
                            **base,
                            "kind": "grid",
                            "clusterer": "kmeans",
                            "hdbscan_min_cluster_size": None,
                            "hdbscan_min_samples": None,
                            "noise_policy": NOISE_NOT_APPLICABLE,
                            "kmeans_k": k,
                            "agglomerative_k": None,
                            "agglomerative_linkage": None,
                        }
                    )
            elif clusterer == "agglomerative":
                for k in agg["k"]:
                    specs.append(
                        {
                            **base,
                            "kind": "grid",
                            "clusterer": "agglomerative",
                            "hdbscan_min_cluster_size": None,
                            "hdbscan_min_samples": None,
                            "noise_policy": NOISE_NOT_APPLICABLE,
                            "kmeans_k": None,
                            "agglomerative_k": k,
                            "agglomerative_linkage": agg.get("linkage", "average"),
                        }
                    )
            else:
                raise ValueError(f"unknown clusterer: {clusterer}")

    valid = [s for s in specs if is_valid_config(s)]
    for spec in valid:
        spec["config_id"] = config_id(spec)
    return valid


def is_valid_config(cfg: Mapping) -> bool:
    """Return True if a run spec is internally consistent.

    Deterministic guard against invalid combinations: clusterer-specific
    parameters must be present only for their own clusterer, and noise_policy
    must be `not_applicable` for clusterers without native noise.
    """
    clusterer = cfg.get("clusterer")
    if clusterer == "hdbscan":
        if cfg.get("hdbscan_min_cluster_size") is None:
            return False
        if cfg.get("hdbscan_min_samples") is None:
            return False
        if cfg.get("noise_policy") in (None, NOISE_NOT_APPLICABLE):
            return False
        if cfg.get("kmeans_k") is not None or cfg.get("agglomerative_k") is not None:
            return False
        return True
    if clusterer == "kmeans":
        if cfg.get("kmeans_k") is None:
            return False
        if cfg.get("noise_policy") != NOISE_NOT_APPLICABLE:
            return False
        if cfg.get("hdbscan_min_cluster_size") is not None:
            return False
        if cfg.get("agglomerative_k") is not None:
            return False
        return True
    if clusterer == "agglomerative":
        if cfg.get("agglomerative_k") is None:
            return False
        if cfg.get("agglomerative_linkage") != "average":
            return False
        if cfg.get("noise_policy") != NOISE_NOT_APPLICABLE:
            return False
        if cfg.get("hdbscan_min_cluster_size") is not None:
            return False
        if cfg.get("kmeans_k") is not None:
            return False
        return True
    return False


def sample_configs(
    configs: Sequence[Dict], max_configs: int, seed: int
) -> List[Dict]:
    """Cap the non-baseline config count using deterministic seeded sampling.

    If `len(configs) <= max_configs` the input is returned unchanged. Otherwise
    `max_configs` configs are drawn with `random.Random(seed)` and returned in
    a stable order (sorted by config_id) so repeated runs are identical.
    """
    if len(configs) <= max_configs:
        return list(configs)
    rng = random.Random(seed)
    chosen = rng.sample(list(configs), max_configs)
    return sorted(chosen, key=lambda c: c["config_id"])


def build_baselines(config: Mapping) -> List[Dict]:
    """Build the fixed HDBSCAN baseline run specs from the config.

    One baseline per `fixed_baselines` entry of the form
    `MERT_<source>_HDBSCAN_default`, using `baseline_definition`.
    """
    definition = config["baseline_definition"]
    source_map = {"perc": "mert_perc", "full": "mert_full", "concat": "mert_concat"}
    baselines: List[Dict] = []
    for name in config["fixed_baselines"]:
        token = name.replace("MERT_", "").replace("_HDBSCAN_default", "").lower()
        source = source_map.get(token)
        if source is None:
            raise ValueError(f"unrecognized fixed baseline name: {name}")
        spec = {
            "kind": "baseline",
            "baseline_name": name,
            "embedding_source": source,
            "mert_aggregation": definition["mert_aggregation"],
            "normalization": definition["normalization"],
            "dim_reduction": definition["dim_reduction"],
            "clusterer": "hdbscan",
            "hdbscan_min_cluster_size": definition["hdbscan_min_cluster_size"],
            "hdbscan_min_samples": definition["hdbscan_min_samples"],
            "noise_policy": definition["noise_policy"],
            "kmeans_k": None,
            "agglomerative_k": None,
            "agglomerative_linkage": None,
        }
        spec["config_id"] = config_id(spec)
        baselines.append(spec)
    return baselines


def build_diagnostic_rows(config: Mapping) -> List[Dict]:
    """Build diagnostic-only reference rows for V2/V4.

    These rows are never executed and never compete for the main score. They
    carry `executable = False` so the runner records them without clustering.
    """
    references = config.get("diagnostic_only_references", list(DIAGNOSTIC_REFERENCES))
    rows: List[Dict] = []
    for ref in references:
        spec = {
            "kind": "diagnostic",
            "reference_name": ref,
            "executable": False,
            "clusterer": None,
            "embedding_source": None,
            "mert_aggregation": None,
            "normalization": None,
            "dim_reduction": None,
            "noise_policy": NOISE_NOT_APPLICABLE,
        }
        spec["config_id"] = config_id(spec)
        rows.append(spec)
    return rows


def assemble_run_plan(config: Mapping) -> Dict:
    """Assemble the full Regime 1 run plan from a loaded sweep config.

    Returns a dict with the capped non-baseline configs, the fixed baselines,
    the diagnostic-only reference rows, and aggregate counts. The fixed
    baselines are exempt from the sampling cap.
    """
    sampling = config["sampling"]
    grid = expand_grid(config)
    sampled = sample_configs(grid, sampling["max_configs"], sampling["seed"])
    baselines = build_baselines(config)
    diagnostics = build_diagnostic_rows(config)
    return {
        "grid_total": len(grid),
        "sampled_configs": sampled,
        "n_sampled": len(sampled),
        "baselines": baselines,
        "n_baselines": len(baselines),
        "diagnostic_rows": diagnostics,
        "n_diagnostic": len(diagnostics),
        "n_executable": len(sampled) + len(baselines),
        "sampling_seed": sampling["seed"],
        "max_configs": sampling["max_configs"],
    }
