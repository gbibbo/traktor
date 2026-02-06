"""
PURPOSE: Generate a tree-style text representation of track organization by genre.
         Reads final_organization.csv and outputs a tree structure similar to the `tree` command.

CHANGELOG:
    2026-02-05: Initial implementation (no external dependencies).
"""
from pathlib import Path
from typing import Dict, List
import argparse
import csv
import sys


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load CSV file using stdlib csv module.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of row dicts
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_tree_structure(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build hierarchical structure from rows.

    Args:
        rows: List of dicts with folder_l1, folder_l2, track keys

    Returns:
        Nested dict: {folder_l1: {folder_l2: [tracks]}}
    """
    tree = {}

    for row in rows:
        l1 = row["folder_l1"]
        l2 = row["folder_l2"]
        track = row["track"]

        if l1 not in tree:
            tree[l1] = {}
        if l2 not in tree[l1]:
            tree[l1][l2] = []
        tree[l1][l2].append(track)

    return tree


def generate_tree_text(tree: Dict[str, Dict[str, List[str]]], title: str = None) -> str:
    """
    Generate tree-style text output.

    Args:
        tree: Hierarchical structure {folder_l1: {folder_l2: [tracks]}}
        title: Optional title for the tree

    Returns:
        Tree as string with box-drawing characters
    """
    lines = []

    if title:
        lines.append(title)

    # Sort folders (put Noise at the end)
    l1_folders = sorted(tree.keys(), key=lambda x: (x == "Noise", x))

    for i, l1 in enumerate(l1_folders):
        is_last_l1 = (i == len(l1_folders) - 1)
        l1_prefix = "└── " if is_last_l1 else "├── "
        l1_indent = "    " if is_last_l1 else "│   "

        # Count tracks in this group
        track_count = sum(len(tracks) for tracks in tree[l1].values())
        lines.append(f"{l1_prefix}{l1} ({track_count} tracks)")

        # Sort subfolders
        l2_folders = sorted(tree[l1].keys(), key=lambda x: (x == "Noise", x))

        for j, l2 in enumerate(l2_folders):
            is_last_l2 = (j == len(l2_folders) - 1)
            l2_prefix = "└── " if is_last_l2 else "├── "
            l2_indent = "    " if is_last_l2 else "│   "

            tracks = sorted(tree[l1][l2])
            lines.append(f"{l1_indent}{l2_prefix}{l2} ({len(tracks)} tracks)")

            for k, track in enumerate(tracks):
                is_last_track = (k == len(tracks) - 1)
                track_prefix = "└── " if is_last_track else "├── "
                lines.append(f"{l1_indent}{l2_indent}{track_prefix}{track}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tree-style text of track organization"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to final_organization.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TRAKTOR ML - Organization by Genre",
        help="Title for the tree"
    )
    args = parser.parse_args()

    # Load CSV
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        return 1

    rows = load_csv(args.input)

    # Validate columns
    if rows:
        required_cols = ["track", "folder_l1", "folder_l2"]
        missing = [c for c in required_cols if c not in rows[0]]
        if missing:
            print(f"[ERROR] Missing columns: {missing}", file=sys.stderr)
            return 1

    print(f"[INFO] Loaded {len(rows)} tracks from {args.input}")

    # Build tree
    tree = build_tree_structure(rows)
    tree_text = generate_tree_text(tree, title=args.title)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(tree_text)
        print(f"[SAVED] {args.output}")
    else:
        print()
        print(tree_text)

    # Stats
    n_groups = len(tree)
    n_subgroups = sum(len(sg) for sg in tree.values())
    print(f"\n[INFO] {n_groups} groups, {n_subgroups} subgroups, {len(rows)} tracks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
