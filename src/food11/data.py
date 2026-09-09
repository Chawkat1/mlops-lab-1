"""
Prepare the food11 dataset for training.

Reads raw images from ./data/food11_raw/{training,evaluation,validation}
and produces two processed copies under ./data:

  - food11_processed:      full dataset, images resized to 128x128,
                            sorted into per-category subfolders.
  - food11_processed_mini: same as above, capped at 100 images per
                            category per split (for fast dev/testing).

Run with:
    uv run python ./src/food11/data.py
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# --- Config ---------------------------------------------------------------

RAW_DIR = Path("backup_data/food11_raw_full")
PROCESSED_DIR = Path("backup_data/food11_processed_full")
PROCESSED_MINI_DIR = Path("backup_data/food11_processed_mini")

SPLITS = ["training", "evaluation", "validation"]

TARGET_SIZE = (128, 128)
MINI_LIMIT_PER_CATEGORY = 100

# food11 (Kaggle karakaggle/food11) numeric-label -> category name mapping.
CATEGORY_NAMES = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles-Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable-Fruit",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --- Helpers ----------------------------------------------------------------


def category_from_filename(filename: str) -> str | None:
    """
    Extract the category label from a food11 filename.

    Files are named like "<label>_<id>.jpg", e.g. "0_10.jpg" -> label "0".
    Returns the human-readable category name, or None if it can't be parsed.
    """
    match = re.match(r"^(\d+)_", filename)
    if not match:
        return None
    label = match.group(1)
    return CATEGORY_NAMES.get(label)


def iter_split_images(split_dir: Path):
    """Yield image files in a split directory."""
    if not split_dir.exists():
        return
    for path in sorted(split_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resize_and_save(src_path: Path, dst_path: Path, size: tuple[int, int]) -> None:
    """Resize an image and save it, converting to RGB to avoid mode issues."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(dst_path)


# --- Main processing ---------------------------------------------------------


def process_split(split: str, mini: bool) -> None:
    """Process a single split (training/evaluation/validation)."""
    src_split_dir = RAW_DIR / split
    out_root = PROCESSED_MINI_DIR if mini else PROCESSED_DIR
    dst_split_dir = out_root / split

    if not src_split_dir.exists():
        print(f"  [skip] {src_split_dir} does not exist")
        return

    counts: dict[str, int] = defaultdict(int)
    written = 0
    skipped_unknown = 0

    for img_path in iter_split_images(src_split_dir):
        category = category_from_filename(img_path.name)
        if category is None:
            skipped_unknown += 1
            continue

        if mini and counts[category] >= MINI_LIMIT_PER_CATEGORY:
            continue

        dst_path = dst_split_dir / category / img_path.name
        resize_and_save(img_path, dst_path, TARGET_SIZE)

        counts[category] += 1
        written += 1

    label = "mini" if mini else "full"
    print(f"  [{label}] {split}: wrote {written} images across {len(counts)} categories")
    if skipped_unknown:
        print(f"  [{label}] {split}: skipped {skipped_unknown} files with unrecognized names")


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(
            f"Raw data folder not found at '{RAW_DIR}'. "
            "Make sure you've run `dvc pull` or placed the raw data there."
        )

    # Start clean so re-running the script doesn't mix stale + new files.
    for out_dir in (PROCESSED_DIR, PROCESSED_MINI_DIR):
        if out_dir.exists():
            shutil.rmtree(out_dir)

    print("Processing full dataset -> food11_processed ...")
    for split in SPLITS:
        process_split(split, mini=False)

    print("Processing mini dataset -> food11_processed_mini ...")
    for split in SPLITS:
        process_split(split, mini=True)

    print("Done.")


if __name__ == "__main__":
    main()
