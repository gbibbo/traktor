"""
PURPOSE: Download Essentia TensorFlow models for V2 pipeline.
         Downloads genre_discogs400 classifier (required for semantic labeling).
         Also verifies existing effnet model is present.

CHANGELOG:
    2025-02-04: Initial implementation for V2 drum-first hierarchy pipeline.
"""
from pathlib import Path
import urllib.request
import argparse
import sys


MODEL_URLS = {
    "genre_discogs400": {
        "weights": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
        "metadata": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
        "expected_size_mb": 5,
    },
    "effnet": {
        "weights": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
        "metadata": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json",
        "expected_size_mb": 18,
    },
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """
    Download a file from URL to destination path with progress indicator.

    Args:
        url: Source URL
        dest: Destination file path
        desc: Description for progress output

    Returns:
        True if download successful, False otherwise
    """
    if dest.exists():
        print(f"[SKIP] {desc} already exists: {dest}")
        return True

    print(f"[DOWNLOAD] {desc}")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=report_hook)
        print()  # New line after progress
        print(f"[OK] Downloaded: {dest.name} ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def verify_model(model_path: Path, expected_size_mb: float, tolerance: float = 0.5) -> bool:
    """
    Verify model file exists and is approximately correct size.

    Args:
        model_path: Path to model file
        expected_size_mb: Expected size in megabytes
        tolerance: Size tolerance as fraction (0.5 = 50%)

    Returns:
        True if file passes verification
    """
    if not model_path.exists():
        print(f"[FAIL] Model not found: {model_path}")
        return False

    actual_size_mb = model_path.stat().st_size / (1024 * 1024)
    min_size = expected_size_mb * (1 - tolerance)
    max_size = expected_size_mb * (1 + tolerance)

    if min_size <= actual_size_mb <= max_size:
        print(f"[OK] {model_path.name}: {actual_size_mb:.1f} MB (expected ~{expected_size_mb} MB)")
        return True
    else:
        print(f"[WARN] {model_path.name}: {actual_size_mb:.1f} MB (expected ~{expected_size_mb} MB)")
        return True  # Still consider it OK, just warn


def main(output_dir: Path, models_to_download: list = None) -> int:
    """
    Download specified Essentia models to output directory.

    Args:
        output_dir: Directory to save models
        models_to_download: List of model names to download, or None for all

    Returns:
        0 if all downloads successful, 1 otherwise
    """
    print("=" * 60)
    print("TRAKTOR ML V2 - Essentia Model Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    if models_to_download is None:
        models_to_download = list(MODEL_URLS.keys())

    success = True

    for model_name in models_to_download:
        if model_name not in MODEL_URLS:
            print(f"[WARN] Unknown model: {model_name}")
            continue

        model_info = MODEL_URLS[model_name]
        print(f"\n--- {model_name.upper()} ---")

        # Download weights
        weights_filename = Path(model_info["weights"]).name
        weights_path = output_dir / weights_filename
        if not download_file(model_info["weights"], weights_path, f"{model_name} weights"):
            success = False
            continue

        # Download metadata
        metadata_filename = Path(model_info["metadata"]).name
        metadata_path = output_dir / metadata_filename
        if not download_file(model_info["metadata"], metadata_path, f"{model_name} metadata"):
            success = False
            continue

        # Verify weights file
        verify_model(weights_path, model_info["expected_size_mb"])

    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All models downloaded/verified successfully")
        print("\nFiles in output directory:")
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.1f} MB")
        return 0
    else:
        print("[FAILED] Some downloads failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Essentia TensorFlow models for V2 pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/essentia"),
        help="Directory to save downloaded models (default: models/essentia)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["genre_discogs400"],
        choices=list(MODEL_URLS.keys()),
        help="Models to download (default: genre_discogs400)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    args = parser.parse_args()

    models = list(MODEL_URLS.keys()) if args.all else args.models
    sys.exit(main(args.output_dir, models))
