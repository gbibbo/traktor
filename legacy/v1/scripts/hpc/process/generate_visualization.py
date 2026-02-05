"""
PURPOSE: Generate interactive Plotly visualization of clustered audio embeddings.
         Creates a scatter plot with hover information showing track names.
         Supports double-click to play tracks from a local folder.

CHANGELOG:
    2026-02-04: Added double-click to play local audio files.
    2025-02-03: Initial implementation for Phase 1 validation.
"""
from pathlib import Path
from typing import List, Optional
import argparse
import json
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_results(csv_path: Path) -> pd.DataFrame:
    """
    Load results CSV with UMAP coordinates and cluster labels.

    Args:
        csv_path: Path to results CSV

    Returns:
        DataFrame with columns: track, cluster, umap_x, umap_y
    """
    df = pd.read_csv(csv_path)
    required_columns = ["track", "cluster", "umap_x", "umap_y"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[INFO] Loaded {len(df)} tracks from {csv_path}")
    return df


def create_scatter_plot(
    df: pd.DataFrame,
    title: str = "Audio Embedding Clusters",
    x_col: str = "umap_x",
    y_col: str = "umap_y",
    color_col: str = "cluster",
    hover_cols: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create interactive scatter plot of embeddings.

    Args:
        df: DataFrame with UMAP coordinates and cluster labels
        title: Plot title
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Column for color coding
        hover_cols: Columns to show on hover

    Returns:
        Plotly Figure object
    """
    if hover_cols is None:
        hover_cols = ["track"]

    # Create cluster label column for display
    df = df.copy()
    df["cluster_label"] = df[color_col].apply(
        lambda x: "Noise" if x == -1 else f"Cluster {x}"
    )

    # Count clusters for statistics
    n_clusters = df[df[color_col] != -1][color_col].nunique()
    n_noise = (df[color_col] == -1).sum()

    # Create color palette
    # Use distinct colors for clusters, gray for noise
    unique_labels = sorted(df["cluster_label"].unique())
    color_map = {}
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set3

    for i, label in enumerate(unique_labels):
        if label == "Noise":
            color_map[label] = "rgba(128, 128, 128, 0.5)"  # Gray with transparency
        else:
            color_map[label] = colors[i % len(colors)]

    # Create figure
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="cluster_label",
        color_discrete_map=color_map,
        hover_data=hover_cols,
        title=title,
    )

    # Customize layout
    fig.update_layout(
        title={
            "text": f"{title}<br><sup>{n_clusters} clusters, {n_noise} noise points, {len(df)} total tracks</sup>",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        legend_title="Cluster",
        template="plotly_white",
        width=1000,
        height=800,
        hovermode="closest",
    )

    # Customize markers
    fig.update_traces(
        marker=dict(
            size=10,
            line=dict(width=1, color="DarkSlateGrey"),
        ),
        selector=dict(mode="markers"),
    )

    # Make hover show track name prominently
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
    )

    return fig


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

    # Create track name to index mapping
    track_map = {name: i for i, name in enumerate(track_names)}

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

        // Add double-click handler
        plotDiv.on('plotly_doubleclick', function(data) {{
            // Plotly doubleclick doesn't give us point info, so we use click with timer
        }});

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
        // Method 1: window.open (may be blocked by browser)
        const newWindow = window.open(fileUrl, '_blank');

        if (!newWindow || newWindow.closed) {{
            // Method 2: Show path to copy
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
                    <h2 style="margin:0 0 15px 0;color:#4CAF50;">🎵 Reproducir Track</h2>
                    <p style="margin:0 0 10px 0;font-size:18px;"><strong>${{trackName}}</strong></p>
                    <p style="margin:0 0 20px 0;color:#aaa;font-size:12px;">El navegador bloqueó la apertura directa. Copia la ruta:</p>
                    <input type="text" value="${{LOCAL_AUDIO_DIR}}${{trackName}}" readonly
                           style="width:100%;padding:12px;border:1px solid #444;border-radius:6px;background:#2d2d2d;color:#fff;font-size:14px;margin-bottom:15px;"
                           onclick="this.select();" id="traktor-path-input">
                    <div style="display:flex;gap:10px;">
                        <button onclick="navigator.clipboard.writeText(document.getElementById('traktor-path-input').value);this.textContent='✓ Copiado!';"
                                style="flex:1;padding:12px;background:#4CAF50;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;">
                            📋 Copiar Ruta
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


def save_html(
    fig: go.Figure,
    output_path: Path,
    include_plotlyjs: str = "cdn",
    local_audio_dir: Optional[str] = None,
    track_names: Optional[List[str]] = None,
) -> Path:
    """
    Save figure as standalone HTML file.

    Args:
        fig: Plotly Figure object
        output_path: Output file path
        include_plotlyjs: How to include Plotly JS ("cdn", True for inline, False for none)
        local_audio_dir: Optional local path to audio folder for double-click playback
        track_names: Optional list of track names for playback feature

    Returns:
        Path to saved file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate base HTML
    html_content = fig.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=True,
    )

    # Inject playback JavaScript if local audio dir is specified
    if local_audio_dir and track_names:
        playback_js = generate_playback_js(local_audio_dir, track_names)
        # Insert before closing </body> tag
        html_content = html_content.replace("</body>", f"{playback_js}</body>")
        print(f"[INFO] Double-click playback enabled for local folder: {local_audio_dir}")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[SAVED] {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive Plotly visualization of clustered embeddings"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to results CSV with track, cluster, umap_x, umap_y columns"
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        required=True,
        help="Path to output HTML file"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TRAKTOR ML - Audio Embedding Clusters",
        help="Plot title"
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "inline"],
        default="cdn",
        help="How to include Plotly JS (cdn = smaller file, inline = fully offline)"
    )
    parser.add_argument(
        "--local-audio-dir",
        type=str,
        default=None,
        help="Local path to audio folder for double-click playback (e.g., 'C:\\\\Música\\\\2020 new - copia')"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRAKTOR ML - Embedding Visualization Generator")
    print("=" * 60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output HTML: {args.output_html}")
    print(f"Title: {args.title}")
    if args.local_audio_dir:
        print(f"Local audio dir: {args.local_audio_dir}")
    print()

    # Load data
    df = load_results(args.input_csv)

    # Create plot
    print("[INFO] Creating scatter plot...")
    fig = create_scatter_plot(
        df,
        title=args.title,
    )

    # Save HTML
    include_js = True if args.include_plotlyjs == "inline" else "cdn"
    track_names = df["track"].tolist() if args.local_audio_dir else None
    save_html(
        fig,
        args.output_html,
        include_plotlyjs=include_js,
        local_audio_dir=args.local_audio_dir,
        track_names=track_names,
    )

    # Print summary
    n_clusters = df[df["cluster"] != -1]["cluster"].nunique()
    n_noise = (df["cluster"] == -1).sum()

    print("\n" + "=" * 60)
    print("[SUCCESS] Visualization generated!")
    print(f"  Tracks: {len(df)}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    print(f"  Output: {args.output_html}")
    if args.local_audio_dir:
        print(f"  Playback: Double-click any point to play")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
