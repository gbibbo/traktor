#!/usr/bin/env python3
"""
PURPOSE: Genera visualization.html desde CSVs intermedios.
         Separado de phase2_analysis.py para iteracion rapida.
         Incluye toggle para alternar entre vista L1 (Drums) y L2 (Todo).

CHANGELOG:
  - v1.0: Creacion inicial con toggle L1/L2
  - v1.1: Eliminada dependencia de pandas, usa solo stdlib
  - v1.2: Boton deseleccionar todos, genero en nombres L2, jerarquia en leyenda

USAGE:
  python scripts/local/generate_visualization.py [results_dir] [--audio-dir PATH]

EXAMPLE:
  python scripts/local/generate_visualization.py results/v2_hierarchy/
  python scripts/local/generate_visualization.py results/v2_hierarchy/ --audio-dir "C:/Música/2020 new - copia/"
"""

import argparse
import colorsys
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any


# =============================================================================
# Color Utilities
# =============================================================================

def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to HSL."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL to hex color."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


def adjust_color_for_subcluster(base_hex: str, sub_index: int, total_subs: int) -> str:
    """Generate a color variation for a subcluster based on parent color."""
    h, s, l = hex_to_hsl(base_hex)

    if total_subs <= 1:
        return base_hex

    # Vary hue slightly and adjust lightness
    hue_shift = (sub_index - 1) * 0.05  # Small hue variation
    h = (h + hue_shift) % 1.0

    # Vary lightness: first subcluster is base, subsequent are lighter/darker
    lightness_shift = (sub_index - 1) * 0.08
    l = max(0.2, min(0.8, l + lightness_shift * (1 if sub_index % 2 == 0 else -1)))

    return hsl_to_hex(h, s, l)


# =============================================================================
# L1 Color Palette (matches original visualization)
# =============================================================================

L1_BASE_COLORS = {
    'Noise': '#808080',
    'A': '#ab63fa',
    'B': '#FFA15A',
    'C': '#EF553B',
    'D': '#FF6692',
    'E': '#19d3f3',
    'F': '#FF97FF',
    'G': '#ab63fa',
    'H': '#EF553B',
    'I': '#636efa',
    'J': '#00cc96',
    'K': '#19d3f3',
    'L': '#FECB52',
    'M': '#FFA15A',
    'N': '#00cc96',
    'O': '#FF6692',
    'P': '#B6E880',
}


def get_l1_color(cluster_l1: str) -> str:
    """Get color for L1 cluster."""
    return L1_BASE_COLORS.get(cluster_l1, '#888888')


def generate_l2_color_palette(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate unique colors for each L2 subcluster."""
    l2_colors = {}

    # Group by L1 first
    l1_groups = group_by(data, 'cluster_l1')

    for l1, rows in l1_groups.items():
        subclusters = sorted(set(r['cluster_l2'] for r in rows))
        base_color = get_l1_color(l1)

        for i, l2 in enumerate(subclusters, 1):
            l2_colors[l2] = adjust_color_for_subcluster(base_color, i, len(subclusters))

    return l2_colors


# =============================================================================
# Data Loading (using stdlib csv module)
# =============================================================================

def read_csv_as_dicts(filepath: Path) -> List[Dict[str, str]]:
    """Read CSV file and return list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_data(results_dir: Path) -> List[Dict[str, Any]]:
    """Load and merge all CSV files into a list of dicts."""
    # Load CSVs
    l1_data = read_csv_as_dicts(results_dir / 'level1_clusters.csv')
    l2_data = read_csv_as_dicts(results_dir / 'level2_clusters.csv')
    org_data = read_csv_as_dicts(results_dir / 'final_organization.csv')

    # Build lookup dicts
    l2_lookup = {row['track']: row.get('cluster_l2', '') for row in l2_data}
    org_lookup = {row['track']: row.get('folder_l1', 'Unknown') for row in org_data}

    # Merge data
    merged = []
    for row in l1_data:
        track = row['track']
        merged.append({
            'track': track,
            'cluster_l1': row['cluster_l1'],
            'umap_x': float(row['umap_x']),
            'umap_y': float(row['umap_y']),
            'cluster_l2': l2_lookup.get(track) or row['cluster_l1'],
            'folder_l1': org_lookup.get(track, 'Unknown'),
        })

    return merged


def build_genre_label_map(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from cluster_l1 to genre label for legend."""
    # Group by cluster_l1
    clusters: Dict[str, List[str]] = {}
    for row in data:
        l1 = row['cluster_l1']
        if l1 not in clusters:
            clusters[l1] = []
        clusters[l1].append(row['folder_l1'])

    genre_map = {}
    for l1, folders in clusters.items():
        # Extract base genre from folder name (before _[)
        genres = [f.split('_[')[0] if '_[' in f else f for f in folders]
        genre_counts = Counter(genres)
        top_genre = genre_counts.most_common(1)[0][0]
        if l1 == 'Noise':
            genre_map[l1] = 'Noise'
        else:
            genre_map[l1] = f"{top_genre} ({l1})"

    return genre_map


def build_l1_genre_map(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from cluster_l1 to just the genre name (without letter)."""
    clusters: Dict[str, List[str]] = {}
    for row in data:
        l1 = row['cluster_l1']
        if l1 not in clusters:
            clusters[l1] = []
        clusters[l1].append(row['folder_l1'])

    genre_map = {}
    for l1, folders in clusters.items():
        genres = [f.split('_[')[0] if '_[' in f else f for f in folders]
        genre_counts = Counter(genres)
        top_genre = genre_counts.most_common(1)[0][0]
        genre_map[l1] = top_genre

    return genre_map


def build_l2_genre_map(data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from cluster_l2 to genre name based on voting within subcluster."""
    clusters: Dict[str, List[str]] = {}
    for row in data:
        l2 = row['cluster_l2']
        if l2 not in clusters:
            clusters[l2] = []
        clusters[l2].append(row['folder_l1'])

    genre_map = {}
    for l2, folders in clusters.items():
        genres = [f.split('_[')[0] if '_[' in f else f for f in folders]
        genre_counts = Counter(genres)
        top_genre = genre_counts.most_common(1)[0][0]
        genre_map[l2] = top_genre

    return genre_map


# =============================================================================
# Trace Building
# =============================================================================

def group_by(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group list of dicts by a key."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in data:
        k = row[key]
        if k not in groups:
            groups[k] = []
        groups[k].append(row)
    return groups


def build_l1_traces(data: List[Dict[str, Any]], genre_map: Dict[str, str]) -> List[dict]:
    """Build Plotly traces for L1 view (grouped by cluster_l1)."""
    traces = []
    groups = group_by(data, 'cluster_l1')

    for l1 in sorted(groups.keys()):
        rows = groups[l1]
        genre_label = genre_map.get(l1, l1)

        trace = {
            'x': [r['umap_x'] for r in rows],
            'y': [r['umap_y'] for r in rows],
            'mode': 'markers',
            'type': 'scatter',
            'name': genre_label,
            'legendgroup': f'L1_{l1}',
            'marker': {
                'color': get_l1_color(l1),
                'size': 8,
                'symbol': 'circle'
            },
            'customdata': [
                [r['track'], r['cluster_l1'], r['cluster_l2'], r['folder_l1']]
                for r in rows
            ],
            'hovertemplate': (
                'genre_label=%{fullData.name}<br>'
                'umap_x=%{x}<br>'
                'umap_y=%{y}<br>'
                'track=%{customdata[0]}<br>'
                'cluster_l1=%{customdata[1]}<br>'
                'cluster_l2=%{customdata[2]}<br>'
                'folder_l1=%{customdata[3]}'
                '<extra></extra>'
            ),
            'visible': True,
            'showlegend': True,
        }
        traces.append(trace)

    return traces


def build_l2_traces(
    data: List[Dict[str, Any]],
    l2_colors: Dict[str, str],
    l1_genre_map: Dict[str, str],
    l2_genre_map: Dict[str, str]
) -> List[dict]:
    """Build Plotly traces for L2 view with hierarchy (grouped by cluster_l2)."""
    traces = []

    # Group data by L1 first, then by L2 within each L1
    l1_groups = group_by(data, 'cluster_l1')

    for l1 in sorted(l1_groups.keys()):
        l1_rows = l1_groups[l1]
        l2_subgroups = group_by(l1_rows, 'cluster_l2')
        l1_genre = l1_genre_map.get(l1, l1)

        # Determine if this is the first trace in the L1 group (for legendgrouptitle)
        is_first_in_group = True

        for l2 in sorted(l2_subgroups.keys()):
            rows = l2_subgroups[l2]
            l2_genre = l2_genre_map.get(l2, '')

            # Build name with genre: "A1 (Techno)"
            display_name = f"{l2} ({l2_genre})" if l2_genre else l2

            trace = {
                'x': [r['umap_x'] for r in rows],
                'y': [r['umap_y'] for r in rows],
                'mode': 'markers',
                'type': 'scatter',
                'name': display_name,
                'legendgroup': f'L1_{l1}',  # Group by parent L1
                'marker': {
                    'color': l2_colors.get(l2, '#888888'),
                    'size': 8,
                    'symbol': 'circle'
                },
                'customdata': [
                    [r['track'], r['cluster_l1'], r['cluster_l2'], r['folder_l1']]
                    for r in rows
                ],
                'hovertemplate': (
                    'cluster_l2=%{fullData.name}<br>'
                    'umap_x=%{x}<br>'
                    'umap_y=%{y}<br>'
                    'track=%{customdata[0]}<br>'
                    'cluster_l1=%{customdata[1]}<br>'
                    'cluster_l2=%{customdata[2]}<br>'
                    'folder_l1=%{customdata[3]}'
                    '<extra></extra>'
                ),
                'visible': False,  # Hidden initially
                'showlegend': True,
            }

            # Add legendgrouptitle only to first trace of each L1 group
            if is_first_in_group:
                trace['legendgrouptitle'] = {
                    'text': f"<b>{l1} - {l1_genre}</b>",
                    'font': {'size': 12, 'color': '#333'}
                }
                is_first_in_group = False

            traces.append(trace)

    return traces


# =============================================================================
# JavaScript Generation
# =============================================================================

def generate_toggle_js(num_l1_traces: int, num_l2_traces: int) -> str:
    """Generate JavaScript for toggle functionality."""
    return f'''
<script>
// TRAKTOR ML - Cluster Level Toggle
(function() {{
    const NUM_L1_TRACES = {num_l1_traces};
    const NUM_L2_TRACES = {num_l2_traces};
    let currentView = 'L1';

    function setClusterView(level) {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) return;

        const l1Indices = Array.from({{length: NUM_L1_TRACES}}, (_, i) => i);
        const l2Indices = Array.from({{length: NUM_L2_TRACES}}, (_, i) => i + NUM_L1_TRACES);

        if (level === 'L1') {{
            // Show L1, hide L2
            Plotly.restyle(plotDiv, {{visible: true}}, l1Indices);
            Plotly.restyle(plotDiv, {{visible: false}}, l2Indices);
        }} else {{
            // Hide L1, show L2
            Plotly.restyle(plotDiv, {{visible: false}}, l1Indices);
            Plotly.restyle(plotDiv, {{visible: true}}, l2Indices);
        }}

        // Update button UI
        document.getElementById('btn-l1').classList.toggle('active', level === 'L1');
        document.getElementById('btn-l2').classList.toggle('active', level === 'L2');
        currentView = level;
    }}

    function deselectAll() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) return;

        const l1Indices = Array.from({{length: NUM_L1_TRACES}}, (_, i) => i);
        const l2Indices = Array.from({{length: NUM_L2_TRACES}}, (_, i) => i + NUM_L1_TRACES);

        if (currentView === 'L1') {{
            // Set all L1 traces to 'legendonly' (visible in legend but not on plot)
            Plotly.restyle(plotDiv, {{visible: 'legendonly'}}, l1Indices);
        }} else {{
            // Set all L2 traces to 'legendonly'
            Plotly.restyle(plotDiv, {{visible: 'legendonly'}}, l2Indices);
        }}
        console.log('[TRAKTOR ML] Deselected all traces in ' + currentView + ' view');
    }}

    function selectAll() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv) return;

        const l1Indices = Array.from({{length: NUM_L1_TRACES}}, (_, i) => i);
        const l2Indices = Array.from({{length: NUM_L2_TRACES}}, (_, i) => i + NUM_L1_TRACES);

        if (currentView === 'L1') {{
            // Show all L1 traces
            Plotly.restyle(plotDiv, {{visible: true}}, l1Indices);
        }} else {{
            // Show all L2 traces
            Plotly.restyle(plotDiv, {{visible: true}}, l2Indices);
        }}
        console.log('[TRAKTOR ML] Selected all traces in ' + currentView + ' view');
    }}

    // Expose to global scope
    window.setClusterView = setClusterView;
    window.deselectAll = deselectAll;
    window.selectAll = selectAll;

    console.log('[TRAKTOR ML] Toggle initialized: ' + NUM_L1_TRACES + ' L1 traces, ' + NUM_L2_TRACES + ' L2 traces');
}})();
</script>
'''


def generate_playback_js(audio_dir: str, track_names: List[str]) -> str:
    """Generate JavaScript for double-click playback."""
    import json
    tracks_json = json.dumps(track_names, ensure_ascii=False)

    return f'''
<script>
// TRAKTOR ML - Double-click to play tracks
(function() {{
    const LOCAL_AUDIO_DIR = "{audio_dir}";
    const TRACK_NAMES = {tracks_json};

    function setupDoubleClick() {{
        // Buscar el div de Plotly por ID o por clase
        let plotDiv = document.getElementById('plotly-graph');
        if (!plotDiv) {{
            plotDiv = document.querySelector('.plotly-graph-div');
        }}

        if (!plotDiv) {{
            console.log('[TRAKTOR ML] Waiting for Plotly div...');
            setTimeout(setupDoubleClick, 200);
            return;
        }}

        // Verificar que Plotly está listo
        if (!plotDiv.on) {{
            console.log('[TRAKTOR ML] Waiting for Plotly to initialize...');
            setTimeout(setupDoubleClick, 200);
            return;
        }}

        let lastClickTime = 0;
        let lastClickedTrack = null;

        plotDiv.on('plotly_click', function(data) {{
            console.log('[TRAKTOR ML] Click detected', data);

            if (!data || !data.points || data.points.length === 0) {{
                console.log('[TRAKTOR ML] No points in click data');
                return;
            }}

            const point = data.points[0];
            console.log('[TRAKTOR ML] Point customdata:', point.customdata);

            const trackName = point.customdata ? point.customdata[0] : null;

            if (!trackName) {{
                console.log('[TRAKTOR ML] No track name found');
                return;
            }}

            const now = Date.now();

            if (trackName === lastClickedTrack && (now - lastClickTime) < 400) {{
                console.log('[TRAKTOR ML] Double-click detected on: ' + trackName);
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
        // Construir URL file:// correctamente con encoding
        const fileUrl = 'file:///' + LOCAL_AUDIO_DIR + encodeURIComponent(trackName).replace(/%2F/g, '/');
        console.log('[TRAKTOR ML] Playing: ' + trackName);
        console.log('[TRAKTOR ML] URL: ' + fileUrl);

        // Intentar abrir con window.open primero
        const newWindow = window.open(fileUrl, '_blank');

        // Si el navegador bloquea file://, mostrar modal como fallback
        if (!newWindow || newWindow.closed || typeof newWindow.closed === 'undefined') {{
            const filePath = LOCAL_AUDIO_DIR + trackName;
            showPlaybackModal(trackName, filePath);
        }}
    }}

    function showPlaybackModal(trackName, filePath) {{
        const existing = document.getElementById('traktor-modal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'traktor-modal';
        modal.innerHTML = `
            <div style="position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);z-index:9999;display:flex;align-items:center;justify-content:center;" onclick="if(event.target===this)document.getElementById('traktor-modal').remove();">
                <div style="background:#1e1e1e;padding:30px;border-radius:12px;max-width:700px;width:90%;color:#fff;font-family:system-ui;" onclick="event.stopPropagation();">
                    <h2 style="margin:0 0 15px 0;color:#4CAF50;">Reproducir Track</h2>
                    <p style="margin:0 0 10px 0;font-size:16px;word-break:break-all;"><strong>${{trackName}}</strong></p>
                    <p style="margin:0 0 15px 0;color:#aaa;font-size:13px;">Copia la ruta y abrela en tu reproductor:</p>
                    <input type="text" value="${{filePath}}" readonly
                           style="width:100%;padding:12px;border:1px solid #444;border-radius:6px;background:#2d2d2d;color:#fff;font-size:13px;margin-bottom:15px;box-sizing:border-box;"
                           onclick="this.select();" id="traktor-path-input">
                    <div style="display:flex;gap:10px;">
                        <button onclick="navigator.clipboard.writeText(document.getElementById('traktor-path-input').value).then(()=>this.textContent='Copiado!').catch(()=>this.textContent='Error');"
                                style="flex:1;padding:12px;background:#4CAF50;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:500;">
                            Copiar Ruta
                        </button>
                        <button onclick="document.getElementById('traktor-modal').remove();"
                                style="flex:1;padding:12px;background:#555;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;">
                            Cerrar
                        </button>
                    </div>
                    <p style="margin:15px 0 0 0;color:#666;font-size:11px;text-align:center;">Presiona Escape o haz click fuera para cerrar</p>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Cerrar con Escape
        const handleEsc = (e) => {{
            if (e.key === 'Escape') {{
                document.getElementById('traktor-modal')?.remove();
                document.removeEventListener('keydown', handleEsc);
            }}
        }};
        document.addEventListener('keydown', handleEsc);

        setTimeout(() => {{
            const input = document.getElementById('traktor-path-input');
            if (input) input.select();
        }}, 100);
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', setupDoubleClick);
    }} else {{
        setupDoubleClick();
    }}
}})();
</script>
'''


# =============================================================================
# HTML Generation
# =============================================================================

def generate_html(
    traces: List[dict],
    num_l1_traces: int,
    num_l2_traces: int,
    track_names: List[str],
    audio_dir: str
) -> str:
    """Generate complete HTML with Plotly visualization and toggle."""
    import json

    traces_json = json.dumps(traces, ensure_ascii=False)

    layout = {
        'title': {'text': 'TRAKTOR ML V2 - Drum-First Hierarchy Clustering'},
        'xaxis': {'title': {'text': 'umap_x'}, 'anchor': 'y', 'domain': [0.0, 1.0]},
        'yaxis': {'title': {'text': 'umap_y'}, 'anchor': 'x', 'domain': [0.0, 1.0]},
        'legend': {
            'title': {'text': 'Cluster'},
            'tracegroupgap': 0,
            'itemsizing': 'constant'
        },
        'width': 1200,
        'height': 800,
        'hovermode': 'closest',
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#E5ECF6',
    }
    layout_json = json.dumps(layout)

    toggle_js = generate_toggle_js(num_l1_traces, num_l2_traces)
    playback_js = generate_playback_js(audio_dir, track_names)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>TRAKTOR ML - Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, sans-serif;
        }}
        #cluster-controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
        }}
        .toggle-container {{
            display: flex;
            background: #2d2d2d;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .toggle-btn {{
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            background: #444;
            color: #aaa;
        }}
        .toggle-btn:hover {{
            background: #555;
        }}
        .toggle-btn.active {{
            background: #4CAF50;
            color: #fff;
        }}
        .toggle-label {{
            padding: 10px 15px;
            background: #1e1e1e;
            color: #888;
            font-size: 12px;
            display: flex;
            align-items: center;
        }}
    </style>
</head>
<body>
    <div id="cluster-controls">
        <div class="toggle-container">
            <span class="toggle-label">Vista:</span>
            <button id="btn-l1" class="toggle-btn active" onclick="setClusterView('L1')">
                Segun Drums
            </button>
            <button id="btn-l2" class="toggle-btn" onclick="setClusterView('L2')">
                Segun Todo
            </button>
            <span class="toggle-label" style="margin-left:10px;">|</span>
            <button id="btn-deselect" class="toggle-btn" onclick="deselectAll()" title="Ocultar todos para ver uno a uno">
                Deseleccionar
            </button>
            <button id="btn-select" class="toggle-btn" onclick="selectAll()" title="Mostrar todos los clusters">
                Seleccionar
            </button>
        </div>
    </div>

    <div>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <div id="plotly-graph" class="plotly-graph-div" style="height:800px; width:1200px;"></div>
        <script type="text/javascript">
            Plotly.newPlot(
                "plotly-graph",
                {traces_json},
                {layout_json},
                {{responsive: true}}
            );
        </script>
    </div>

{toggle_js}
{playback_js}
</body>
</html>
'''
    return html


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization.html from CSV files'
    )
    parser.add_argument(
        'results_dir',
        type=Path,
        nargs='?',
        default=Path('results/v2_hierarchy'),
        help='Directory containing CSV files (default: results/v2_hierarchy)'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='C:/Música/2020 new - copia/',
        help='Local audio directory for playback'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output HTML path (default: results_dir/visualization.html)'
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or (results_dir / 'visualization.html')

    print(f"Loading data from {results_dir}...")
    data = load_data(results_dir)
    print(f"  Loaded {len(data)} tracks")

    l1_clusters = sorted(set(r['cluster_l1'] for r in data))
    l2_clusters = sorted(set(r['cluster_l2'] for r in data))
    print(f"  L1 clusters: {l1_clusters}")
    print(f"  L2 clusters: {l2_clusters}")

    # Build genre label maps
    genre_map = build_genre_label_map(data)
    l1_genre_map = build_l1_genre_map(data)
    l2_genre_map = build_l2_genre_map(data)
    print(f"  L1 Genre labels: {genre_map}")
    print(f"  L2 Genre labels: {l2_genre_map}")

    # Generate L2 color palette
    l2_colors = generate_l2_color_palette(data)
    print(f"  Generated {len(l2_colors)} L2 colors")

    # Build traces
    print("Building traces...")
    l1_traces = build_l1_traces(data, genre_map)
    l2_traces = build_l2_traces(data, l2_colors, l1_genre_map, l2_genre_map)
    all_traces = l1_traces + l2_traces
    print(f"  L1 traces: {len(l1_traces)}")
    print(f"  L2 traces: {len(l2_traces)}")

    # Generate HTML
    print("Generating HTML...")
    track_names = [r['track'] for r in data]
    html = generate_html(
        traces=all_traces,
        num_l1_traces=len(l1_traces),
        num_l2_traces=len(l2_traces),
        track_names=track_names,
        audio_dir=args.audio_dir
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Saved to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
