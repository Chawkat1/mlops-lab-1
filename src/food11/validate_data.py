"""
Check a processed food11 dataset against the data contract.

Usage:
    uv run python src/food11/validate_data.py data/food11_processed_mini
    uv run python src/food11/validate_data.py data/food11_processed --min-per-class 200

Exits with status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

CLASSES = {
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
}
SPLITS = ["training", "validation", "evaluation"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed food11 dataset")
    parser.add_argument("dataset", type=Path, help="e.g. data/food11_processed_mini")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def report(name: str, problems: list[str]) -> bool:
    ok = not problems
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for problem in problems[:5]:
        print(f"         {problem}")
    if len(problems) > 5:
        print(f"         ... and {len(problems) - 5} more")
    return ok


def check_structure(root: Path) -> list[str]:
    problems = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            problems.append(f"{split}: folder is missing")
            continue
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if found != CLASSES:
            missing = sorted(CLASSES - found)
            extra = sorted(found - CLASSES)
            problems.append(f"{split}: missing={missing} extra={extra}")
    return problems


def check_images(images: dict[str, dict[str, list[Path]]]) -> list[str]:
    problems = []
    for split in images.values():
        for paths in split.values():
            for path in paths:
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != (128, 128):
                            problems.append(f"{path}: mode={img.mode} size={img.size}")
                except Exception as exc:
                    problems.append(f"{path}: cannot open ({exc})")
    return problems


def check_min_per_class(images: dict[str, dict[str, list[Path]]], minimum: int) -> list[str]:
    problems = []
    for split, classes in images.items():
        for cls, paths in classes.items():
            if len(paths) < minimum:
                problems.append(f"{split}/{cls}: {len(paths)} images (< {minimum})")
    return problems


def check_no_leakage(images: dict[str, dict[str, list[Path]]]) -> list[str]:
    seen: dict[str, tuple[str, Path]] = {}
    problems = []
    for split, classes in images.items():
        for paths in classes.values():
            for path in paths:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                if digest in seen and seen[digest][0] != split:
                    problems.append(f"{path} is the same file as {seen[digest][1]}")
                seen.setdefault(digest, (split, path))
    return problems


def check_class_balance(images: dict[str, dict[str, list[Path]]], tau: float) -> list[str]:
    def shares(split: str) -> dict[str, float]:
        total = sum(len(v) for v in images[split].values())
        return {cls: len(v) / total for cls, v in images[split].items()}

    train, val = shares("training"), shares("validation")
    problems = []
    for cls in sorted(train):
        gap = abs(train[cls] - val.get(cls, 0.0))
        if gap > tau:
            problems.append(f"{cls}: train {train[cls]:.3f} vs validation {val.get(cls, 0.0):.3f}")
    return problems


def main() -> None:
    args = parse_args()
    root = args.dataset

    results = []
    structure_problems = check_structure(root)
    results.append(report("each split has exactly the 11 class folders", structure_problems))

    if structure_problems:
        # Later checks need a sane folder layout to make any sense.
        print("Stopping here: fix the folder layout first.")
        sys.exit(1)

    images = {
        split: {cls.name: list_images(cls) for cls in sorted((root / split).iterdir()) if cls.is_dir()}
        for split in SPLITS
    }

    results.append(report("all images open, are RGB and 128x128", check_images(images)))
    results.append(
        report(
            f"every class has at least {args.min_per_class} images per split",
            check_min_per_class(images, args.min_per_class),
        )
    )
    results.append(report("no image appears in two splits", check_no_leakage(images)))
    results.append(
        report(
            f"class shares in training and validation differ by at most {args.tau}",
            check_class_balance(images, args.tau),
        )
    )

    if not all(results):
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
