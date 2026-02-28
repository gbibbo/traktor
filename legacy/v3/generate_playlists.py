#!/usr/bin/env python3
# No mover a Legacy
"""
PURPOSE: Generate Traktor-compatible playlists (M3U) from clustering results.
         Creates one playlist per cluster with versioned output folders (V1, V2, V3...).

CHANGELOG:
    2026-02-04: Changed from M3U8 to M3U format for Traktor compatibility.
    2026-02-04: Initial implementation.

USAGE:
    python generate_playlists.py [--results-csv PATH] [--local-audio-dir PATH]

OUTPUT:
    playlists/V{N}/
        ├── Cluster_00.m3u
        ├── Cluster_01.m3u
        ├── ...
        ├── Cluster_15.m3u
        └── Noise.m3u
"""
from pathlib import Path
import argparse
import csv
import re
import sys
from typing import Dict, List


# Default paths
DEFAULT_RESULTS_CSV = Path(__file__).parent / "results" / "full_collection" / "results.csv"
DEFAULT_LOCAL_AUDIO_DIR = r"C:\Música\2020 new - copia"
PLAYLISTS_BASE_DIR = Path(__file__).parent / "playlists"


def get_next_version_folder(base_dir: Path) -> Path:
    """
    Find the next available version folder (V1, V2, V3, ...).

    Args:
        base_dir: Base directory for playlists

    Returns:
        Path to the next version folder
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find existing version folders
    existing_versions = []
    for item in base_dir.iterdir():
        if item.is_dir() and re.match(r'^V(\d+)$', item.name):
            version_num = int(re.match(r'^V(\d+)$', item.name).group(1))
            existing_versions.append(version_num)

    # Determine next version
    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        next_version = 1

    return base_dir / f"V{next_version}"


def load_clusters(csv_path: Path) -> Dict[str, List[str]]:
    """
    Load clustering results and group tracks by cluster.

    Args:
        csv_path: Path to results.csv

    Returns:
        Dictionary mapping cluster names to list of track filenames
    """
    clusters: Dict[str, List[str]] = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            track = row['track']
            cluster_id = int(row['cluster'])

            # Create cluster name
            if cluster_id == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster_{cluster_id:02d}"

            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(track)

    return clusters


def generate_m3u_playlist(
    tracks: List[str],
    playlist_name: str,
    local_audio_dir: str,
    output_path: Path
) -> None:
    """
    Generate an M3U playlist file compatible with Traktor.

    Args:
        tracks: List of track filenames
        playlist_name: Name of the playlist
        local_audio_dir: Local Windows path to audio folder
        output_path: Path to output .m3u file
    """
    # Normalize the audio directory path
    audio_dir = local_audio_dir.rstrip('\\').rstrip('/')

    with open(output_path, 'w', encoding='utf-8') as f:
        # M3U header
        f.write('#EXTM3U\n')
        f.write(f'#PLAYLIST:{playlist_name}\n')
        f.write('\n')

        for track in sorted(tracks):
            # Write extended info (optional, but nice for display)
            # Format: #EXTINF:duration,Artist - Title
            # We don't have duration, so use -1
            track_display = track.replace('.mp3', '').replace('.wav', '').replace('.flac', '')
            f.write(f'#EXTINF:-1,{track_display}\n')

            # Write the full path (Windows format)
            full_path = f'{audio_dir}\\{track}'
            f.write(f'{full_path}\n')
            f.write('\n')

    print(f"  [OK] {output_path.name}: {len(tracks)} tracks")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Traktor-compatible playlists from clustering results"
    )
    parser.add_argument(
        '--results-csv',
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Path to results.csv (default: {DEFAULT_RESULTS_CSV})"
    )
    parser.add_argument(
        '--local-audio-dir',
        type=str,
        default=DEFAULT_LOCAL_AUDIO_DIR,
        help=f"Local Windows path to audio folder (default: {DEFAULT_LOCAL_AUDIO_DIR})"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help="Override output directory (skips version auto-increment)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TRAKTOR ML - Playlist Generator")
    print("=" * 60)
    print(f"Results CSV: {args.results_csv}")
    print(f"Local audio dir: {args.local_audio_dir}")
    print()

    # Check if results file exists
    if not args.results_csv.exists():
        print(f"[ERROR] Results file not found: {args.results_csv}")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_next_version_folder(PLAYLISTS_BASE_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load clusters
    print("[INFO] Loading clustering results...")
    clusters = load_clusters(args.results_csv)
    print(f"[INFO] Found {len(clusters)} clusters")
    print()

    # Generate playlists
    print("[INFO] Generating playlists...")
    total_tracks = 0

    for cluster_name in sorted(clusters.keys()):
        tracks = clusters[cluster_name]
        total_tracks += len(tracks)

        # Generate playlist filename
        playlist_file = output_dir / f"{cluster_name}.m3u"

        # Generate the playlist
        generate_m3u_playlist(
            tracks=tracks,
            playlist_name=f"TRAKTOR ML - {cluster_name}",
            local_audio_dir=args.local_audio_dir,
            output_path=playlist_file
        )

    # Summary
    print()
    print("=" * 60)
    print("[SUCCESS] Playlists generated!")
    print(f"  Total playlists: {len(clusters)}")
    print(f"  Total tracks: {total_tracks}")
    print(f"  Output: {output_dir}")
    print()
    print("To import in Traktor:")
    print("  1. Copy the playlists folder to your Windows machine")
    print("  2. In Traktor: File > Import Collection...")
    print("     Or drag & drop .m3u files into the playlist panel")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
