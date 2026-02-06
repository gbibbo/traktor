"""
PURPOSE: Phase 2 of V2 pipeline - Clustering, genre classification, and report generation.
         Performs two-level clustering (drums -> fulltrack) and semantic labeling.

CHANGELOG:
    2026-02-05: Show genre names in visualization legend and folder names (strip Electronic--- prefix).
    2026-02-05: Added double-click playback to visualization (ported from legacy).
    2025-02-04: Initial implementation for V2 drum-first hierarchy pipeline.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import Counter
import argparse
import json
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.common.clustering_utils import (
    l2_normalize,
    apply_umap,
    apply_hdbscan,
    cluster_to_letter,
    letter_to_cluster,
    subcluster_label,
    get_cluster_stats,
    simplify_genre_name,
)
from scripts.common.audio_utils import get_audio_files, load_audio_essentia
from scripts.common.embedding_utils import extract_genre_predictions


def load_embeddings_v2(embeddings_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load V2 embeddings and manifest from directory.

    Args:
        embeddings_dir: Directory containing embedding files

    Returns:
        drum_embeddings: Shape (N, 1280)
        fulltrack_embeddings: Shape (N, 1280)
        filenames: List of track names
    """
    # Load manifest
    manifest_path = embeddings_dir / "manifest_v2.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    filenames = manifest["tracks"]

    # Load embeddings
    drum_path = embeddings_dir / "drum_embeddings.npy"
    fulltrack_path = embeddings_dir / "fulltrack_embeddings.npy"

    drum_embeddings = np.load(drum_path)
    fulltrack_embeddings = np.load(fulltrack_path)

    print(f"[INFO] Loaded drum embeddings: {drum_embeddings.shape}")
    print(f"[INFO] Loaded fulltrack embeddings: {fulltrack_embeddings.shape}")
    print(f"[INFO] Number of tracks: {len(filenames)}")

    return drum_embeddings, fulltrack_embeddings, filenames


def cluster_level1(
    drum_embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Level 1 clustering on drum embeddings.

    Args:
        drum_embeddings: Shape (N, D)
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        min_cluster_size: HDBSCAN parameter
        min_samples: HDBSCAN parameter

    Returns:
        cluster_labels: Shape (N,) numeric labels
        umap_coords: Shape (N, 2) UMAP coordinates
    """
    print("\n" + "=" * 40)
    print("LEVEL 1 CLUSTERING (Drums)")
    print("=" * 40)

    # Normalize
    normalized = l2_normalize(drum_embeddings)

    # UMAP
    umap_coords = apply_umap(
        normalized,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    # HDBSCAN
    labels = apply_hdbscan(
        umap_coords,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    stats = get_cluster_stats(labels)
    print(f"[LEVEL1] Clusters: {stats['n_clusters']}, Noise: {stats['n_noise']}")

    return labels, umap_coords


def cluster_level2(
    fulltrack_embeddings: np.ndarray,
    level1_labels: np.ndarray,
    min_tracks_for_subcluster: int = 10,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Perform Level 2 sub-clustering within each Level 1 cluster.

    Args:
        fulltrack_embeddings: Shape (N, D)
        level1_labels: Shape (N,) Level 1 cluster labels
        min_tracks_for_subcluster: Minimum tracks needed to sub-cluster
        n_neighbors: UMAP parameter for subclustering
        min_dist: UMAP parameter for subclustering
        min_cluster_size: HDBSCAN parameter for subclustering
        min_samples: HDBSCAN parameter for subclustering

    Returns:
        level2_labels: Shape (N,) combined labels like "A1", "A2", "B1", etc.
        sub_umap_coords: Dict mapping cluster letter to UMAP coords
    """
    print("\n" + "=" * 40)
    print("LEVEL 2 CLUSTERING (Fulltrack)")
    print("=" * 40)

    n_tracks = len(level1_labels)
    level2_labels = [""] * n_tracks
    sub_umap_coords = {}

    unique_clusters = sorted(set(level1_labels))

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            # Noise points get "Noise" label
            for i in range(n_tracks):
                if level1_labels[i] == -1:
                    level2_labels[i] = "Noise"
            continue

        cluster_letter = cluster_to_letter(cluster_id)
        cluster_mask = level1_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        print(f"\n[LEVEL2] Cluster {cluster_letter}: {cluster_size} tracks")

        if cluster_size < min_tracks_for_subcluster:
            # Too few tracks - assign all to single subcluster
            print(f"  -> Too few tracks for subclustering, assigning to {cluster_letter}1")
            for idx in cluster_indices:
                level2_labels[idx] = subcluster_label(cluster_letter, 0)
            sub_umap_coords[cluster_letter] = None
            continue

        # Get fulltrack embeddings for this cluster
        cluster_embeddings = fulltrack_embeddings[cluster_mask]
        normalized = l2_normalize(cluster_embeddings)

        # Sub-cluster
        try:
            sub_umap = apply_umap(
                normalized,
                n_neighbors=min(n_neighbors, cluster_size - 1),
                min_dist=min_dist,
                verbose=False,
            )
            sub_labels = apply_hdbscan(
                sub_umap,
                min_cluster_size=min(min_cluster_size, cluster_size // 3 + 1),
                min_samples=min(min_samples, cluster_size // 4 + 1),
                verbose=False,
            )

            sub_umap_coords[cluster_letter] = sub_umap

            # Assign subclusters
            sub_stats = get_cluster_stats(sub_labels)
            print(f"  -> Subclusters: {sub_stats['n_clusters']}, Noise: {sub_stats['n_noise']}")

            for local_idx, global_idx in enumerate(cluster_indices):
                sub_id = sub_labels[local_idx]
                if sub_id == -1:
                    level2_labels[global_idx] = f"{cluster_letter}_noise"
                else:
                    level2_labels[global_idx] = subcluster_label(cluster_letter, sub_id)

        except Exception as e:
            print(f"  -> Subclustering failed: {e}, assigning to {cluster_letter}1")
            for idx in cluster_indices:
                level2_labels[idx] = subcluster_label(cluster_letter, 0)
            sub_umap_coords[cluster_letter] = None

    return level2_labels, sub_umap_coords


def classify_genres(
    audio_dir: Path,
    filenames: List[str],
    effnet_model_path: Path,
    genre_model_path: Path,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Classify genres for all tracks.

    Args:
        audio_dir: Directory containing audio files
        filenames: List of filenames to classify
        effnet_model_path: Path to effnet model
        genre_model_path: Path to genre classifier model
        top_n: Number of top genres to return per track

    Returns:
        DataFrame with columns: track, genre_1, genre_2, genre_3, conf_1, conf_2, conf_3
    """
    print("\n" + "=" * 40)
    print("GENRE CLASSIFICATION")
    print("=" * 40)

    results = []

    for filename in tqdm(filenames, desc="Classifying genres"):
        audio_path = audio_dir / filename

        if not audio_path.exists():
            print(f"\n[WARN] Audio not found: {filename}")
            row = {"track": filename}
            for i in range(top_n):
                row[f"genre_{i+1}"] = "unknown"
                row[f"conf_{i+1}"] = 0.0
            results.append(row)
            continue

        try:
            audio = load_audio_essentia(audio_path)
            genres, confidences = extract_genre_predictions(
                audio, effnet_model_path, genre_model_path, top_n
            )

            row = {"track": filename}
            for i in range(top_n):
                if i < len(genres):
                    row[f"genre_{i+1}"] = genres[i]
                    row[f"conf_{i+1}"] = confidences[i]
                else:
                    row[f"genre_{i+1}"] = ""
                    row[f"conf_{i+1}"] = 0.0
            results.append(row)

        except Exception as e:
            print(f"\n[WARN] Genre classification failed for {filename}: {e}")
            row = {"track": filename}
            for i in range(top_n):
                row[f"genre_{i+1}"] = "error"
                row[f"conf_{i+1}"] = 0.0
            results.append(row)

    df = pd.DataFrame(results)
    print(f"[INFO] Classified {len(df)} tracks")

    return df


def generate_folder_names(
    level1_labels: np.ndarray,
    level2_labels: List[str],
    genre_df: pd.DataFrame,
    top_n_genres: int = 2,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate semantic folder names based on genre voting.

    Args:
        level1_labels: Numeric Level 1 cluster labels
        level2_labels: String Level 2 labels like "A1", "B2"
        genre_df: DataFrame with genre predictions
        top_n_genres: Number of top genres to include in folder name

    Returns:
        l1_folder_names: Dict mapping cluster letter to folder name
        l2_folder_names: Dict mapping subcluster label to folder name
    """
    print("\n" + "=" * 40)
    print("GENERATING FOLDER NAMES")
    print("=" * 40)

    n_tracks = len(level1_labels)
    l1_folder_names = {}
    l2_folder_names = {}

    # Level 1 folder names
    unique_l1 = sorted(set(level1_labels))
    for cluster_id in unique_l1:
        if cluster_id == -1:
            l1_folder_names["Noise"] = "Noise"
            continue

        cluster_letter = cluster_to_letter(cluster_id)
        cluster_mask = level1_labels == cluster_id

        # Get genres for this cluster
        cluster_genres = []
        for i, is_in_cluster in enumerate(cluster_mask):
            if is_in_cluster and i < len(genre_df):
                g1 = genre_df.iloc[i].get("genre_1", "")
                if g1 and g1 not in ["unknown", "error"]:
                    cluster_genres.append(g1)

        # Vote
        if cluster_genres:
            genre_counts = Counter(cluster_genres)
            top_genres = [g for g, c in genre_counts.most_common(top_n_genres)]
            simplified_genres = [simplify_genre_name(g) for g in top_genres]
            genre_str = "_".join(simplified_genres)
            primary_genre = simplified_genres[0] if simplified_genres else cluster_letter
            folder_name = f"{primary_genre}_[{genre_str}]"
        else:
            folder_name = f"Group_{cluster_letter}"

        l1_folder_names[cluster_letter] = folder_name
        print(f"  {cluster_letter} -> {folder_name}")

    # Level 2 folder names
    unique_l2 = sorted(set(level2_labels))
    for l2_label in unique_l2:
        if l2_label == "Noise" or "_noise" in l2_label:
            l2_folder_names[l2_label] = l2_label
            continue

        # Get tracks in this subcluster
        l2_mask = [l == l2_label for l in level2_labels]

        sub_genres = []
        for i, is_in_sub in enumerate(l2_mask):
            if is_in_sub and i < len(genre_df):
                g1 = genre_df.iloc[i].get("genre_1", "")
                if g1 and g1 not in ["unknown", "error"]:
                    sub_genres.append(g1)

        if sub_genres:
            genre_counts = Counter(sub_genres)
            top_genre = genre_counts.most_common(1)[0][0]
            folder_name = simplify_genre_name(top_genre)
        else:
            folder_name = l2_label

        l2_folder_names[l2_label] = folder_name

    return l1_folder_names, l2_folder_names


def generate_playback_js(local_audio_dir: str, track_names: List[str]) -> str:
    """
    Generate JavaScript code for double-click track playback.

    Args:
        local_audio_dir: Local Windows path to audio folder (e.g., "C:\\Música\\2020 new - copia")
        track_names: List of track filenames

    Returns:
        JavaScript code as string
    """
    # Normalize path for JavaScript (use forward slashes for file:// URLs)
    audio_dir_normalized = local_audio_dir.replace("\\", "/")
    if not audio_dir_normalized.endswith("/"):
        audio_dir_normalized += "/"

    js_code = f"""
<script>
// TRAKTOR ML - Double-click to play tracks
(function() {{
    const LOCAL_AUDIO_DIR = "{audio_dir_normalized}";
    const TRACK_NAMES = {json.dumps(track_names)};

    // Wait for Plotly to be ready
    function setupDoubleClick() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) {{
            setTimeout(setupDoubleClick, 100);
            return;
        }}

        let lastClickTime = 0;
        let lastClickedTrack = null;

        plotDiv.on('plotly_click', function(data) {{
            if (!data || !data.points || data.points.length === 0) return;

            const point = data.points[0];
            const trackName = point.customdata ? point.customdata[0] : null;

            if (!trackName) return;

            const now = Date.now();

            // Detect double-click (within 400ms)
            if (trackName === lastClickedTrack && (now - lastClickTime) < 400) {{
                playTrack(trackName);
                lastClickTime = 0;
                lastClickedTrack = null;
            }} else {{
                lastClickTime = now;
                lastClickedTrack = trackName;
            }}
        }});

        console.log('[TRAKTOR ML] Double-click playback enabled for ' + TRACK_NAMES.length + ' tracks');
    }}

    function playTrack(trackName) {{
        // Build file:// URL
        const fileUrl = 'file:///' + LOCAL_AUDIO_DIR + encodeURIComponent(trackName).replace(/%2F/g, '/');

        console.log('[TRAKTOR ML] Opening: ' + trackName);

        // Try to open the file
        const newWindow = window.open(fileUrl, '_blank');

        if (!newWindow || newWindow.closed) {{
            // Show path to copy
            showPlaybackModal(trackName, fileUrl);
        }}
    }}

    function showPlaybackModal(trackName, fileUrl) {{
        // Remove existing modal
        const existing = document.getElementById('traktor-modal');
        if (existing) existing.remove();

        // Create modal
        const modal = document.createElement('div');
        modal.id = 'traktor-modal';
        modal.innerHTML = `
            <div style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:9999;display:flex;align-items:center;justify-content:center;">
                <div style="background:#1e1e1e;padding:30px;border-radius:12px;max-width:600px;color:#fff;font-family:system-ui;">
                    <h2 style="margin:0 0 15px 0;color:#4CAF50;">Reproducir Track</h2>
                    <p style="margin:0 0 10px 0;font-size:18px;"><strong>${{trackName}}</strong></p>
                    <p style="margin:0 0 20px 0;color:#aaa;font-size:12px;">El navegador bloqueó la apertura directa. Copia la ruta:</p>
                    <input type="text" value="${{LOCAL_AUDIO_DIR}}${{trackName}}" readonly
                           style="width:100%;padding:12px;border:1px solid #444;border-radius:6px;background:#2d2d2d;color:#fff;font-size:14px;margin-bottom:15px;"
                           onclick="this.select();" id="traktor-path-input">
                    <div style="display:flex;gap:10px;">
                        <button onclick="navigator.clipboard.writeText(document.getElementById('traktor-path-input').value);this.textContent='Copiado!';"
                                style="flex:1;padding:12px;background:#4CAF50;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;">
                            Copiar Ruta
                        </button>
                        <button onclick="document.getElementById('traktor-modal').remove();"
                                style="flex:1;padding:12px;background:#666;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;">
                            Cerrar
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Close on background click
        modal.querySelector('div').addEventListener('click', function(e) {{
            if (e.target === this) modal.remove();
        }});

        // Select the path
        setTimeout(() => document.getElementById('traktor-path-input').select(), 100);
    }}

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', setupDoubleClick);
    }} else {{
        setupDoubleClick();
    }}
}})();
</script>
"""
    return js_code


def generate_visualization(
    output_dir: Path,
    filenames: List[str],
    umap_coords: np.ndarray,
    level1_labels: np.ndarray,
    level2_labels: List[str],
    l1_folder_names: Dict[str, str],
    genre_df: pd.DataFrame,
    local_audio_dir: Optional[str] = None,
) -> Path:
    """
    Generate interactive Plotly visualization.

    Args:
        output_dir: Directory to save HTML
        filenames: Track filenames
        umap_coords: UMAP coordinates
        level1_labels: Level 1 cluster labels
        level2_labels: Level 2 cluster labels
        l1_folder_names: Level 1 folder names
        genre_df: DataFrame with genre predictions
        local_audio_dir: Optional local path to audio folder for double-click playback

    Returns:
        Path to saved HTML file
    """
    import plotly.express as px
    import plotly.graph_objects as go

    print("\n[INFO] Generating visualization...")

    # Build cluster letter list
    cluster_letters = [cluster_to_letter(l) for l in level1_labels]
    unique_letters = sorted(set(cluster_letters))

    # Build genre label map for each cluster
    genre_label_map = {}
    for letter in unique_letters:
        if letter == "Noise":
            genre_label_map[letter] = "Noise"
            continue

        cluster_id = letter_to_cluster(letter)
        cluster_mask = level1_labels == cluster_id

        # Collect genres for this cluster
        cluster_genres = []
        for i, is_in_cluster in enumerate(cluster_mask):
            if is_in_cluster and i < len(genre_df):
                g1 = genre_df.iloc[i].get("genre_1", "")
                if g1 and g1 not in ["unknown", "error"]:
                    cluster_genres.append(g1)

        if cluster_genres:
            genre_counts = Counter(cluster_genres)
            top_genre = genre_counts.most_common(1)[0][0]
            genre_label_map[letter] = simplify_genre_name(top_genre)
        else:
            genre_label_map[letter] = letter

    # Disambiguate duplicate genre labels by appending cluster letter
    genre_counts = Counter(genre_label_map.values())
    for letter, genre in list(genre_label_map.items()):
        if genre_counts[genre] > 1 and genre != "Noise":
            genre_label_map[letter] = f"{genre} ({letter})"

    # Build dataframe
    df = pd.DataFrame({
        "track": filenames,
        "umap_x": umap_coords[:, 0],
        "umap_y": umap_coords[:, 1],
        "cluster_l1": cluster_letters,
        "cluster_l2": level2_labels,
        "genre_label": [genre_label_map[cl] for cl in cluster_letters],
    })

    # Add folder names
    df["folder_l1"] = df["cluster_l1"].map(l1_folder_names)

    # Create scatter plot with customdata for playback
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="genre_label",
        hover_data=["track", "cluster_l1", "cluster_l2", "folder_l1"],
        title="TRAKTOR ML V2 - Drum-First Hierarchy Clustering",
        custom_data=["track"],
    )

    fig.update_layout(
        width=1200,
        height=800,
        hovermode="closest",
    )

    # Generate base HTML
    html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)

    # Inject playback JavaScript if local audio dir is specified
    if local_audio_dir:
        playback_js = generate_playback_js(local_audio_dir, filenames)
        html_content = html_content.replace("</body>", f"{playback_js}</body>")
        print(f"[INFO] Double-click playback enabled for: {local_audio_dir}")

    # Save
    html_path = output_dir / "visualization.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SAVED] {html_path}")

    return html_path


def save_results(
    output_dir: Path,
    filenames: List[str],
    level1_labels: np.ndarray,
    level2_labels: List[str],
    umap_coords: np.ndarray,
    genre_df: pd.DataFrame,
    l1_folder_names: Dict[str, str],
    l2_folder_names: Dict[str, str],
) -> None:
    """
    Save all results to CSV files.

    Args:
        output_dir: Directory to save results
        filenames: Track filenames
        level1_labels: Level 1 cluster labels
        level2_labels: Level 2 cluster labels
        umap_coords: UMAP coordinates
        genre_df: Genre predictions DataFrame
        l1_folder_names: Level 1 folder names
        l2_folder_names: Level 2 folder names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Level 1 clusters CSV
    l1_df = pd.DataFrame({
        "track": filenames,
        "cluster_l1": [cluster_to_letter(l) for l in level1_labels],
        "umap_x": umap_coords[:, 0],
        "umap_y": umap_coords[:, 1],
    })
    l1_path = output_dir / "level1_clusters.csv"
    l1_df.to_csv(l1_path, index=False)
    print(f"[SAVED] {l1_path}")

    # Level 2 clusters CSV
    l2_df = pd.DataFrame({
        "track": filenames,
        "cluster_l1": [cluster_to_letter(l) for l in level1_labels],
        "cluster_l2": level2_labels,
    })
    l2_path = output_dir / "level2_clusters.csv"
    l2_df.to_csv(l2_path, index=False)
    print(f"[SAVED] {l2_path}")

    # Genre predictions CSV
    genre_path = output_dir / "genre_predictions.csv"
    genre_df.to_csv(genre_path, index=False)
    print(f"[SAVED] {genre_path}")

    # Final organization CSV
    final_df = pd.DataFrame({
        "track": filenames,
        "cluster_l1": [cluster_to_letter(l) for l in level1_labels],
        "cluster_l2": level2_labels,
        "folder_l1": [l1_folder_names.get(cluster_to_letter(l), "Unknown") for l in level1_labels],
        "folder_l2": [l2_folder_names.get(l2, l2) for l2 in level2_labels],
    })
    final_path = output_dir / "final_organization.csv"
    final_df.to_csv(final_path, index=False)
    print(f"[SAVED] {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Clustering, genre classification, and report generation"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory containing V2 embeddings"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing audio files (for genre classification)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--effnet-model",
        type=Path,
        default=Path("models/essentia/discogs-effnet-bs64-1.pb"),
        help="Path to Essentia EffNet model"
    )
    parser.add_argument(
        "--genre-model",
        type=Path,
        default=Path("models/essentia/genre_discogs400-discogs-effnet-1.pb"),
        help="Path to genre classifier model"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size for Level 1 (default: 5)"
    )
    parser.add_argument(
        "--min-subcluster-size",
        type=int,
        default=3,
        help="HDBSCAN min_cluster_size for Level 2 (default: 3)"
    )
    parser.add_argument(
        "--skip-genres",
        action="store_true",
        help="Skip genre classification (faster, but no semantic labels)"
    )
    parser.add_argument(
        "--local-audio-dir",
        type=str,
        default=None,
        help="Local path to audio folder for double-click playback in visualization (e.g., 'C:\\\\Music\\\\Collection')"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRAKTOR ML V2 - Phase 2: Analysis & Classification")
    print("=" * 60)
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load embeddings
    drum_emb, full_emb, filenames = load_embeddings_v2(args.embeddings_dir)

    # Level 1 clustering (drums)
    l1_labels, umap_coords = cluster_level1(
        drum_emb,
        min_cluster_size=args.min_cluster_size,
    )

    # Level 2 clustering (fulltrack per L1 cluster)
    l2_labels, sub_umap = cluster_level2(
        full_emb,
        l1_labels,
        min_cluster_size=args.min_subcluster_size,
    )

    # Genre classification
    if args.skip_genres or not args.genre_model.exists():
        if not args.skip_genres:
            print(f"\n[WARN] Genre model not found: {args.genre_model}")
            print("[HINT] Run: python scripts/hpc/process/v2/download_models.py")
        print("[INFO] Skipping genre classification")

        # Create dummy genre df
        genre_df = pd.DataFrame({
            "track": filenames,
            "genre_1": ["unknown"] * len(filenames),
            "genre_2": [""] * len(filenames),
            "genre_3": [""] * len(filenames),
            "conf_1": [0.0] * len(filenames),
            "conf_2": [0.0] * len(filenames),
            "conf_3": [0.0] * len(filenames),
        })
    else:
        genre_df = classify_genres(
            args.audio_dir,
            filenames,
            args.effnet_model,
            args.genre_model,
        )

    # Generate folder names
    l1_folder_names, l2_folder_names = generate_folder_names(
        l1_labels, l2_labels, genre_df
    )

    # Save results
    save_results(
        args.output_dir,
        filenames,
        l1_labels,
        l2_labels,
        umap_coords,
        genre_df,
        l1_folder_names,
        l2_folder_names,
    )

    # Generate visualization
    try:
        generate_visualization(
            args.output_dir,
            filenames,
            umap_coords,
            l1_labels,
            l2_labels,
            l1_folder_names,
            genre_df,
            local_audio_dir=args.local_audio_dir,
        )
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Phase 2 complete!")
    print(f"  Tracks analyzed: {len(filenames)}")
    print(f"  Level 1 clusters: {len(set(l1_labels)) - (1 if -1 in l1_labels else 0)}")
    print(f"  Level 2 subclusters: {len(set(l2_labels))}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
