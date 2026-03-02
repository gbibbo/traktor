"""
PURPOSE: Streamlit dashboard para exploración interactiva del clustering V4.
         Scatter UMAP interactivo, filtros por cluster L1/L2, re-clustering local,
         y botón de export de playlists.
CHANGELOG:
  - 2026-03-01: Creación inicial V4.
  - 2026-03-01: load_data() prefiere parquet con UMAP real; re-cluster pasa --pca-dim y do_umap=True por defecto.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    print(f"[ERROR] Dependencias faltantes: {e}")
    print("Instalar: pip install streamlit plotly")
    sys.exit(1)

from src.v4.common.config_loader import load_config
from src.v4.common.path_resolver import resolve_dataset_artifacts

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TRAKTOR ML V4 — Clustering Explorer",
    page_icon="🎵",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(dataset_name: str):
    """Carga clustering results, catalog y BPM/key. Cacheado por dataset."""
    config = load_config()
    artifacts_dir = resolve_dataset_artifacts(dataset_name, config)
    clustering_dir = artifacts_dir / "clustering"

    # Auto-detect último resultado de clustering con UMAP real (no ceros)
    candidates = sorted(clustering_dir.glob("results_*.parquet"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None, None, None, None

    # Preferir el más reciente que tenga coordenadas UMAP reales (no ceros)
    results_path = candidates[-1]  # fallback: el más reciente
    for p in reversed(candidates):
        df_check = pd.read_parquet(p, columns=["umap_x"])
        if df_check["umap_x"].abs().sum() > 0:
            results_path = p
            break
    config_hash = results_path.stem.replace("results_", "")

    df = pd.read_parquet(results_path)

    # BPM/key
    bpm_path = artifacts_dir / "features" / "bpm_key.parquet"
    bpm_df = pd.read_parquet(bpm_path) if bpm_path.exists() else pd.DataFrame()
    if not bpm_df.empty:
        df = df.merge(bpm_df, on="track_uid", how="left")

    # Catalog success
    cs_path = artifacts_dir / "catalog_success.parquet"
    if cs_path.exists():
        cat = pd.read_parquet(cs_path)
        cols = [c for c in ("artist", "title", "filename") if c in cat.columns]
        df = df.merge(cat[["track_uid"] + cols], on="track_uid", how="left")

    # N canónico
    uids_path = artifacts_dir / "embeddings" / "track_uids.json"
    N = len(json.load(open(uids_path))) if uids_path.exists() else len(df)

    return df, config_hash, N, config


def _has_umap(df: pd.DataFrame) -> bool:
    """True si UMAP coords son significativas (no todos ceros)."""
    if "umap_x" not in df.columns or "umap_y" not in df.columns:
        return False
    return df["umap_x"].abs().sum() > 0 or df["umap_y"].abs().sum() > 0


def _track_label(row: pd.Series) -> str:
    artist = row.get("artist", None)
    title = row.get("title", None)
    if pd.notna(artist) and pd.notna(title):
        return f"{artist} - {title}"
    return str(row.get("filename", row.get("track_uid", "?")))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("TRAKTOR ML V4 — Clustering Explorer")

    # Sidebar: controles globales
    with st.sidebar:
        st.header("Dataset")
        dataset_name = st.text_input("Dataset name", value="test_20")

        st.divider()
        st.header("Filtros")
        filter_l1 = st.selectbox("Filtrar por cluster L1", options=["Todos"])
        show_noise = st.checkbox("Mostrar noise (label=-1)", value=True)

        st.divider()
        st.header("Re-clustering (local)")
        st.warning(
            "⚠️ Corre localmente, NO en HPC. "
            "Puede tardar 1-5 min dependiendo del dataset."
        )
        l1_min_cluster_size = st.slider("L1 min_cluster_size", 3, 30, 10)
        l1_min_samples = st.slider("L1 min_samples", 1, 10, 3)
        l2_min_cluster_size = st.slider("L2 min_cluster_size", 2, 15, 4)
        do_umap = st.checkbox("Incluir UMAP (más lento)", value=True)
        recluster_btn = st.button("🔄 Re-cluster", type="primary")

        st.divider()
        st.header("Export")
        windows_dir = st.text_input(
            "Windows audio dir",
            value=r"C:\Música\2020 new - copia"
        )
        export_btn = st.button("📂 Export Playlists (Phase 3+4+5)")

    # Cargar datos
    df, config_hash, N, config = load_data(dataset_name)

    if df is None:
        st.error(
            f"No se encontraron resultados de clustering para **{dataset_name}**. "
            "Ejecuta primero `phase2_cluster.py`."
        )
        return

    st.caption(f"Config hash: `{config_hash}` | N tracks: {N}")

    # Filtrar L1
    l1_options = sorted([l for l in df["label_l1"].unique()])
    l1_labels_named = {l: (f"Cluster {l}" if l >= 0 else "Noise") for l in l1_options}
    filter_options = ["Todos"] + [l1_labels_named[l] for l in l1_options if l >= 0]

    with st.sidebar:
        filter_l1 = st.selectbox("Filtrar por cluster L1", options=filter_options, key="filter_l1_select")

    selected_l1 = None
    if filter_l1 != "Todos":
        for l, name in l1_labels_named.items():
            if name == filter_l1:
                selected_l1 = l
                break

    df_view = df.copy()
    if not show_noise:
        df_view = df_view[df_view["label_l1"] != -1]
    if selected_l1 is not None:
        df_view = df_view[df_view["label_l1"] == selected_l1]

    # === Scatter UMAP ===
    st.subheader("Mapa de embeddings (UMAP)")

    has_umap = _has_umap(df_view)
    if not has_umap:
        st.info(
            "UMAP no disponible (se ejecutó con `--skip-umap`). "
            "Mostrando BPM vs label_l1 como placeholder visual."
        )
        df_view = df_view.copy()
        df_view["umap_x"] = df_view.get("bpm", 0)
        df_view["umap_y"] = df_view["label_l1"]

    color_col = "label_l2" if selected_l1 is not None else "label_l1"
    df_view["_label"] = df_view[color_col].astype(str)
    df_view["_hover_name"] = df_view.apply(_track_label, axis=1)
    hover_data = {"_hover_name": True, "label_l1": True, "label_l2": True}
    if "bpm" in df_view.columns:
        hover_data["bpm"] = True
    if "key" in df_view.columns:
        hover_data["key"] = True

    fig = px.scatter(
        df_view,
        x="umap_x",
        y="umap_y",
        color="_label",
        hover_name="_hover_name",
        hover_data=hover_data,
        title=f"Embedding space — coloreado por {'L2' if selected_l1 is not None else 'L1'}",
        height=550,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(legend_title=color_col)
    st.plotly_chart(fig, use_container_width=True)

    # === Stats ===
    has_raw = "label_l1_raw" in df.columns
    n_clusters_l1 = len([l for l in df["label_l1"].unique() if l >= 0])
    noise_rate = float((df["label_l1"] == -1).sum() / len(df))
    cols = st.columns(5 if has_raw else 4)
    cols[0].metric("N tracks", N)
    cols[1].metric("L1 Clusters", n_clusters_l1)
    cols[2].metric("Noise rate", f"{noise_rate:.1%}")
    if has_raw:
        raw_noise = float((df["label_l1_raw"] == -1).sum() / len(df))
        cols[3].metric("Noise original", f"{raw_noise:.1%}")
        bpm_col = cols[4]
    else:
        bpm_col = cols[3]
    if "bpm" in df.columns:
        bpm_med = df["bpm"].median()
        bpm_col.metric("BPM median", f"{bpm_med:.1f}")

    # === Cluster breakdown ===
    st.subheader("Distribución de clusters")
    cluster_counts = (
        df[df["label_l1"] >= 0]
        .groupby("label_l1")
        .size()
        .reset_index(name="n_tracks")
        .rename(columns={"label_l1": "L1 cluster"})
    )
    if not cluster_counts.empty:
        fig2 = px.bar(cluster_counts, x="L1 cluster", y="n_tracks",
                      title="Tracks por cluster L1", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # BPM histogram
    if "bpm" in df.columns:
        st.subheader("Distribución BPM")
        fig3 = px.histogram(df, x="bpm", nbins=40, height=250,
                            title="BPM distribution")
        st.plotly_chart(fig3, use_container_width=True)

    # === Re-clustering ===
    if recluster_btn:
        st.subheader("Re-clustering en progreso...")
        log_area = st.empty()
        pca_dim = (config or {}).get("clustering", {}).get("pca_dim", 50)
        cmd = [
            sys.executable, str(REPO_ROOT / "src/v4/pipeline/phase2_cluster.py"),
            "--dataset-name", dataset_name,
            "--l1-min-cluster-size", str(l1_min_cluster_size),
            "--l1-min-samples", str(l1_min_samples),
            "--l2-min-cluster-size", str(l2_min_cluster_size),
            "--l2-min-samples", "2",
            "--pca-dim", str(pca_dim),
        ]
        if not do_umap:
            cmd.append("--skip-umap")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, cwd=str(REPO_ROOT))
            output_lines = []
            for line in proc.stdout:
                output_lines.append(line.rstrip())
                log_area.text_area("Log de re-clustering", "\n".join(output_lines[-30:]), height=200)
            proc.wait()
            if proc.returncode == 0:
                st.success("Re-clustering completado. Recarga la página para ver los nuevos resultados.")
                st.cache_data.clear()
            else:
                st.error("Re-clustering falló. Ver log arriba.")
        except Exception as e:
            st.error(f"Error al ejecutar re-clustering: {e}")

    # === Export ===
    if export_btn:
        st.subheader("Exportando playlists...")
        log_area2 = st.empty()
        for phase_script in ["phase3_name.py", "phase4_order.py"]:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "src/v4/pipeline" / phase_script),
                "--dataset-name", dataset_name,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
                if result.returncode != 0:
                    st.error(f"{phase_script} falló:\n{result.stderr}")
                    break
            except Exception as e:
                st.error(f"Error en {phase_script}: {e}")
                break

        cmd_export = [
            sys.executable,
            str(REPO_ROOT / "src/v4/pipeline/phase5_export.py"),
            "--dataset-name", dataset_name,
            "--windows-audio-dir", windows_dir,
        ]
        try:
            proc2 = subprocess.Popen(cmd_export, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, cwd=str(REPO_ROOT))
            out_lines = []
            for line in proc2.stdout:
                out_lines.append(line.rstrip())
                log_area2.text_area("Log de export", "\n".join(out_lines[-20:]), height=150)
            proc2.wait()
            if proc2.returncode == 0:
                st.success("Playlists exportadas correctamente. Ver carpeta `playlists/`.")
            else:
                st.error("Export falló. Ver log arriba.")
        except Exception as e:
            st.error(f"Error en phase5_export: {e}")


if __name__ == "__main__":
    main()
