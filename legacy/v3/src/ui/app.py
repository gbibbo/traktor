#!/usr/bin/env python3
"""
PURPOSE: Streamlit interactive UI for exploring TRAKTOR ML V3 clustering results.
         Provides real-time parameter tuning, 2D scatter visualization,
         quality metrics, and per-cluster track browsing.

CHANGELOG:
    2026-02-19: Fix local audio playback; add backend diagnostic logging.
    2026-02-06: Initial implementation.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Project root & imports from src.clustering
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.run import load_data  # noqa: E402
from src.clustering.flat import ALGORITHMS, create_clusterer  # noqa: E402
from src.clustering.interface import compute_metrics  # noqa: E402
from src.clustering.hierarchical import HierarchicalClusterer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "dataset"
ALGORITHM_DISPLAY = {
    "kmeans": "KMeans",
    "agglomerative": "Agglomerative",
    "hdbscan": "HDBSCAN",
}
LINKAGE_OPTIONS = ["ward", "complete", "average", "single"]
DEFAULT_AUDIO_DIR = r"C:\Música\2020 new - copia"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_datasets() -> List[str]:
    """Return dataset names that contain X_pca128.npy."""
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in ARTIFACTS_DIR.iterdir()
        if d.is_dir() and (d / "X_pca128.npy").exists()
    )


def format_duration(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _cluster_sort_key(label: str) -> Tuple[int, str]:
    """Sort: numeric first, then alphabetic, then noise last."""
    if label in ("-1", "Noise"):
        return (2, label)
    try:
        return (0, f"{int(label):06d}")
    except ValueError:
        return (1, label)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def cached_load_data(dataset_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load features and manifest, cached by dataset path."""
    X, manifest_df = load_data(Path(dataset_path))
    return X, manifest_df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_algorithm_controls(algo: str, prefix: str) -> Dict[str, Any]:
    """Render dynamic sliders for a given algorithm. Returns kwargs dict."""
    params: Dict[str, Any] = {}
    if algo == "kmeans":
        params["n_clusters"] = st.sidebar.slider(
            f"n_clusters ({prefix})", 2, 100, 5, key=f"{prefix}_k"
        )
        params["random_state"] = 42
    elif algo == "agglomerative":
        params["n_clusters"] = st.sidebar.slider(
            f"n_clusters ({prefix})", 2, 100, 5, key=f"{prefix}_agg_k"
        )
        params["linkage"] = st.sidebar.selectbox(
            f"Linkage ({prefix})", LINKAGE_OPTIONS, key=f"{prefix}_linkage"
        )
    elif algo == "hdbscan":
        params["min_cluster_size"] = st.sidebar.slider(
            f"min_cluster_size ({prefix})", 2, 50, 5, key=f"{prefix}_mcs"
        )
        params["min_samples"] = st.sidebar.slider(
            f"min_samples ({prefix})", 1, 30, 3, key=f"{prefix}_ms"
        )
    return params


def render_sidebar() -> Optional[Dict[str, Any]]:
    """Render sidebar controls. Returns config dict or None if no datasets."""
    st.sidebar.title("TRAKTOR ML Explorer")
    st.sidebar.markdown("---")

    # Dataset selector
    datasets = discover_datasets()
    if not datasets:
        st.sidebar.error("No datasets found in artifacts/dataset/")
        return None

    dataset_name = st.sidebar.selectbox("Dataset", datasets)
    st.sidebar.markdown("---")

    # Mode selector
    mode = st.sidebar.radio("Mode", ["Flat", "Hierarchical"])
    mode_key = mode.lower()
    st.sidebar.markdown("---")

    config: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_dir": str(ARTIFACTS_DIR / dataset_name),
        "mode": mode_key,
    }

    algo_names = list(ALGORITHM_DISPLAY.keys())

    if mode_key == "flat":
        algo = st.sidebar.selectbox(
            "Algorithm", algo_names,
            format_func=lambda x: ALGORITHM_DISPLAY[x],
        )
        config["algorithm"] = algo
        config["algo_params"] = _render_algorithm_controls(algo, "flat")
    else:
        # Hierarchical: L1 + L2
        st.sidebar.subheader("L1 (Coarse)")
        l1_algo = st.sidebar.selectbox(
            "L1 Algorithm", algo_names,
            format_func=lambda x: ALGORITHM_DISPLAY[x],
            key="l1_algo",
        )
        config["l1_algorithm"] = l1_algo
        config["l1_params"] = _render_algorithm_controls(l1_algo, "l1")

        st.sidebar.subheader("L2 (Fine)")
        l2_algo = st.sidebar.selectbox(
            "L2 Algorithm", algo_names,
            format_func=lambda x: ALGORITHM_DISPLAY[x],
            key="l2_algo",
        )
        config["l2_algorithm"] = l2_algo
        config["l2_params"] = _render_algorithm_controls(l2_algo, "l2")

        config["l2_min_points"] = st.sidebar.slider(
            "L2 min points", 2, 50, 10, key="l2_min_pts"
        )

    st.sidebar.markdown("---")
    config["run_clicked"] = st.sidebar.button(
        "Run Clustering", type="primary", width="stretch"
    )

    # Audio playback settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Audio Playback")
    audio_dir = st.sidebar.text_input(
        "Local Audio Base Dir", value=DEFAULT_AUDIO_DIR, key="audio_dir"
    )
    config["audio_dir"] = audio_dir

    return config


# ---------------------------------------------------------------------------
# Clustering execution
# ---------------------------------------------------------------------------
def run_flat_clustering(
    X: np.ndarray, config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run flat clustering, return (labels, metrics)."""
    clusterer = create_clusterer(config["algorithm"], **config["algo_params"])
    labels = clusterer.fit_predict(X)
    metrics = compute_metrics(X, labels)
    return labels, metrics


def run_hierarchical_clustering(
    X: np.ndarray, config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run hierarchical clustering, return (l1_labels, composite_labels, metadata)."""
    hc = HierarchicalClusterer(
        l1_algorithm=config["l1_algorithm"],
        l1_params=config["l1_params"],
        l2_algorithm=config["l2_algorithm"],
        l2_params=config["l2_params"],
        l2_min_points=config["l2_min_points"],
    )
    l1_labels, composite_labels, metadata = hc.fit_predict(X)
    return l1_labels, composite_labels, metadata


# ---------------------------------------------------------------------------
# Visualisation: metrics panel
# ---------------------------------------------------------------------------
def render_metrics_panel(metrics: Dict[str, Any]) -> None:
    """Render a row of metric cards."""
    cols = st.columns(4)

    sil = metrics.get("silhouette")
    cols[0].metric("Silhouette Score", f"{sil:.4f}" if sil is not None else "N/A")
    cols[1].metric("Clusters", metrics.get("n_clusters", 0))
    cols[2].metric("Total Tracks", metrics.get("n_total", 0))

    n_noise = metrics.get("n_noise", 0)
    if n_noise > 0:
        cols[3].metric("Noise Points", n_noise)
    else:
        ch = metrics.get("calinski_harabasz")
        cols[3].metric(
            "Calinski-Harabasz", f"{ch:.2f}" if ch is not None else "N/A"
        )


# ---------------------------------------------------------------------------
# Visualisation: scatter plot
# ---------------------------------------------------------------------------
def render_scatter_plot(
    X: np.ndarray,
    labels: np.ndarray,
    manifest_df: pd.DataFrame,
) -> None:
    """Interactive Plotly scatter coloured by cluster label."""
    plot_df = pd.DataFrame({
        "PC1": X[:, 0],
        "PC2": X[:, 1],
        "Cluster": labels.astype(str),
        "Track": manifest_df["relative_path"].values,
        "Duration": manifest_df["duration"].apply(format_duration).values,
    })

    # Deterministic colour map: noise → gray, clusters → qualitative palette
    unique_labels = sorted(plot_df["Cluster"].unique(), key=_cluster_sort_key)
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    color_map: Dict[str, str] = {}
    ci = 0
    for lbl in unique_labels:
        if lbl in ("-1", "Noise"):
            color_map[lbl] = "rgba(180,180,180,0.5)"
        else:
            color_map[lbl] = palette[ci % len(palette)]
            ci += 1

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        color_discrete_map=color_map,
        category_orders={"Cluster": unique_labels},
        hover_data={
            "Track": True,
            "Duration": True,
            "PC1": ":.3f",
            "PC2": ":.3f",
        },
        title="Cluster Visualization (PCA dim 0 vs dim 1)",
        height=600,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(legend_title_text="Cluster")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Visualisation: track explorer
# ---------------------------------------------------------------------------
def render_track_explorer(
    labels: np.ndarray, manifest_df: pd.DataFrame, audio_dir: str
) -> None:
    """Per-cluster expandable tables with track details and audio playback."""
    st.subheader("Track Explorer")

    df = manifest_df.copy()
    df["Cluster"] = labels.astype(str)
    df["Duration_fmt"] = df["duration"].apply(format_duration)

    unique_labels = sorted(df["Cluster"].unique(), key=_cluster_sort_key)

    for lbl in unique_labels:
        cluster_df = df[df["Cluster"] == lbl].reset_index(drop=True)
        n = len(cluster_df)
        if lbl in ("-1", "Noise"):
            header = f"Noise ({n} tracks)"
        else:
            header = f"Cluster {lbl} ({n} tracks)"

        with st.expander(header, expanded=False):
            st.caption("👇 Selecciona una pista en el menú desplegable de abajo para escucharla")
            # Summary table
            display_df = cluster_df[["relative_path", "Duration_fmt"]].rename(
                columns={"relative_path": "Track", "Duration_fmt": "Duration"}
            )
            st.dataframe(display_df, hide_index=True, width="stretch")

            # Track selector + audio player
            track_names = cluster_df["relative_path"].tolist()
            selected = st.selectbox(
                "Select track to play",
                track_names,
                key=f"track_sel_{lbl}",
            )
            if selected is not None:
                row = cluster_df[cluster_df["relative_path"] == selected].iloc[0]
                audio_path = Path(audio_dir) / row["relative_path"]
                track_id = row.get("track_id", "N/A")
                print(
                    f"[AUDIO] Seleccionado: {track_id} "
                    f"| Ruta: {audio_path} "
                    f"| Existe: {audio_path.exists()}"
                )
                if audio_path.exists():
                    st.audio(str(audio_path))
                else:
                    st.error(f"Archivo no encontrado localmente: {audio_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="TRAKTOR ML — Clustering Explorer",
        page_icon=":headphones:",
        layout="wide",
    )

    # Sidebar
    config = render_sidebar()
    if config is None:
        st.warning("No datasets available. Run the pipeline first.")
        return

    # Load data (cached)
    try:
        X, manifest_df = cached_load_data(config["dataset_dir"])
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Failed to load dataset: {exc}")
        return

    # Header
    st.title("TRAKTOR ML — Clustering Explorer")
    st.caption(
        f"Dataset: **{config['dataset_name']}** | "
        f"{X.shape[0]} tracks | {X.shape[1]}D features"
    )

    # Run clustering on button click
    if config["run_clicked"]:
        try:
            with st.spinner("Running clustering..."):
                if config["mode"] == "flat":
                    labels, metrics = run_flat_clustering(X, config)
                    st.session_state["results"] = {
                        "mode": "flat",
                        "labels": labels,
                        "metrics": metrics,
                    }
                else:
                    l1_labels, composite_labels, metadata = (
                        run_hierarchical_clustering(X, config)
                    )
                    st.session_state["results"] = {
                        "mode": "hierarchical",
                        "l1_labels": l1_labels,
                        "composite_labels": composite_labels,
                        "metadata": metadata,
                        "metrics": metadata.get("l1", {}).get("metrics", {}),
                    }
        except Exception as exc:
            st.error(f"Clustering failed: {exc}")
            return

    # Display results
    if "results" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Clustering**.")
        return

    results = st.session_state["results"]
    if results["mode"] == "flat":
        labels = results["labels"]
    else:
        labels = results["composite_labels"]
    metrics = results["metrics"]

    render_metrics_panel(metrics)
    st.markdown("---")
    render_scatter_plot(X, labels, manifest_df)
    st.markdown("---")
    render_track_explorer(labels, manifest_df, config.get("audio_dir", ""))


if __name__ == "__main__":
    main()
