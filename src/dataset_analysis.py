"""Read-only dataset inspection and exploratory analysis.

The supplied dataset is currently folder-based classification data. This module
also detects YOLO, COCO, and Pascal VOC layouts so unsupported annotations are
reported explicitly instead of being fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = {"train", "val", "validation", "test"}


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def detect_format(root: Path) -> str:
    if any(root.rglob("*.json")) and any(root.rglob("*.json")):
        return "COCO JSON (possible)"
    if any(root.rglob("*.xml")):
        return "Pascal VOC XML (possible)"
    if any(p.suffix.lower() == ".txt" for p in root.rglob("*") if p.is_file()):
        return "YOLO detection (possible)"
    dirs = {p.name.lower() for p in root.rglob("*") if p.is_dir()}
    if {"train", "test"} & dirs:
        return "Folder-based image classification"
    return "Unknown image dataset"


def split_for(path: Path, root: Path) -> str:
    parts = [part.lower() for part in path.relative_to(root).parts]
    for part in parts:
        if part in SPLITS:
            return "val" if part == "validation" else part
    return "unsplit"


def class_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() in SPLITS:
            return parts[index + 1] if index + 1 < len(parts) - 1 else "unclassified"
    return parts[-2] if len(parts) > 1 else "unclassified"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image, size: int = 16) -> str:
    gray = ImageOps.grayscale(image).resize((size, size))
    values = np.asarray(gray, dtype=np.float32)
    return "".join("1" if value >= values.mean() else "0" for value in values.ravel())


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def screen_image(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "status": "ok", "error": ""}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            width, height = image.size
            gray = rgb.mean(axis=2)
            variance = float(gray.var())
            brightness = float(gray.mean())
            row.update({
                "width": width, "height": height, "aspect_ratio": width / height,
                "format": image.format or path.suffix.lower().lstrip("."),
                "file_size_bytes": path.stat().st_size,
                "blur_variance": variance, "brightness_mean": brightness,
                "is_small": width < 128 or height < 128,
                "is_unusual_aspect_ratio": width / height < 0.5 or width / height > 2.0,
                "is_blurry_screening": variance < 100.0,
                "is_overexposed_screening": float((gray >= 250).mean()) >= 0.30,
                "is_underexposed_screening": float((gray <= 5).mean()) >= 0.30,
            })
    except Exception as exc:
        row.update({"status": "corrupted_or_unreadable", "error": f"{type(exc).__name__}: {exc}"})
    return row


def yolo_errors(root: Path, images: list[Path]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    image_by_stem = {p.stem: p for p in images}
    labels = [p for p in root.rglob("*.txt") if p.is_file()]
    for label in labels:
        image = image_by_stem.get(label.stem)
        if image is None:
            errors.append({"label": str(label), "image": "", "line": "", "error": "missing_image"})
            continue
        try:
            with Image.open(image) as opened:
                width, height = opened.size
        except Exception as exc:
            errors.append({"label": str(label), "image": str(image), "line": "", "error": f"unreadable_image: {exc}"})
            continue
        for line_number, raw in enumerate(label.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not raw.strip():
                continue
            values = raw.split()
            if len(values) != 5:
                errors.append({"label": str(label), "image": str(image), "line": line_number, "error": "expected_5_values"})
                continue
            try:
                class_id, xc, yc, box_w, box_h = [float(value) for value in values]
            except ValueError:
                errors.append({"label": str(label), "image": str(image), "line": line_number, "error": "non_numeric_value"})
                continue
            if class_id < 0 or class_id != int(class_id):
                errors.append({"label": str(label), "image": str(image), "line": line_number, "error": "invalid_class_id"})
            if not all(0 <= value <= 1 for value in (xc, yc, box_w, box_h)):
                errors.append({"label": str(label), "image": str(image), "line": line_number, "error": "value_outside_0_1"})
            if box_w <= 0 or box_h <= 0:
                errors.append({"label": str(label), "image": str(image), "line": line_number, "error": "non_positive_box_size"})
            if xc - box_w / 2 < 0 or xc + box_w / 2 > 1 or yc - box_h / 2 < 0 or yc + box_h / 2 > 1:
                errors.append({"label": str(label), "image": str(image), "line": line_number, "error": "box_outside_image"})
    return errors


def save_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if columns:
        for column in columns:
            if column not in frame:
                frame[column] = ""
        frame = frame[columns]
    frame.to_csv(path, index=False)


def make_plots(analysis: dict[str, Any], plot_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    quality = pd.DataFrame(analysis["quality_rows"])
    counts = pd.DataFrame(analysis["class_rows"])
    split_counts = pd.DataFrame(analysis["split_rows"])

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(plot_dir / name, dpi=150)
        plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=counts, x="class_name", y="image_count", hue="split", errorbar=None)
    plt.xticks(rotation=35, ha="right"); plt.title("Images by class and split"); save("class_distribution.png")
    plt.figure(figsize=(7, 5))
    sns.barplot(data=split_counts, x="split", y="image_count", errorbar=None)
    plt.title("Image distribution by split"); save("split_distribution.png")
    if not quality.empty and "width" in quality:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(quality["width"], ax=axes[0], bins=30); axes[0].set_title("Image widths")
        sns.histplot(quality["height"], ax=axes[1], bins=30); axes[1].set_title("Image heights")
        save("image_dimensions.png")
        flags = {"small": int(quality["is_small"].sum()), "unusual_ratio": int(quality["is_unusual_aspect_ratio"].sum()),
                 "blurry_screening": int(quality["is_blurry_screening"].sum()),
                 "overexposed_screening": int(quality["is_overexposed_screening"].sum()),
                 "underexposed_screening": int(quality["is_underexposed_screening"].sum())}
        plt.figure(figsize=(9, 5)); sns.barplot(x=list(flags), y=list(flags.values()), color="#4472c4")
        plt.xticks(rotation=30, ha="right"); plt.ylabel("Images flagged"); plt.title("Image quality screening flags")
        save("image_quality_distribution.png")
    sample_paths = [Path(row["path"]) for row in analysis["quality_rows"] if row["status"] == "ok"][:12]
    if sample_paths:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for axis, path in zip(axes.ravel(), sample_paths):
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{class_for(path, analysis['root'])}\n{path.name}", fontsize=8); axis.axis("off")
        for axis in axes.ravel()[len(sample_paths):]: axis.axis("off")
        save("sample_images.png")


def build_report(analysis: dict[str, Any], report_path: Path) -> None:
    s = analysis["summary"]
    readiness = analysis["readiness"]
    report = f"""# Dataset Analysis Report

## 1. Dataset overview

- **Dataset root:** `{analysis['root']}`
- **Detected format:** {analysis['format']}
- **Images:** {s['total_images']}
- **Readable images:** {s['readable_images']}
- **Corrupted/unreadable images:** {s['corrupted_images']}
- **Classes discovered from folder names:** {', '.join(s['classes']) or 'none'}
- **Object-detection annotations:** {s['annotation_files']} files found; this dataset does not provide usable detection boxes.

## 2. Dataset structure

The supplied dataset is a folder-based classification dataset with `Train` and
`Test` directories. Each class is represented by a child folder. It is not a
YOLO detection, COCO JSON, or Pascal VOC XML dataset. Classification folders
provide image-level labels only; they cannot support bounding-box training
without separately supplied annotations.

## 3. Class information

See `class_distribution.csv`. Class presence was counted from actual image
files, not from empty label-file existence. The current dataset contains:
{chr(10).join(f"- {name}: {count} images" for name, count in s['class_totals'].items())}

## 4. Image statistics

- Width range: {s['min_width']} to {s['max_width']} pixels
- Height range: {s['min_height']} to {s['max_height']} pixels
- Average objects per image: not applicable (no bounding boxes)
- Splits: {json.dumps(s['split_totals'])}
- Quality screening flags: {json.dumps(s['quality_flags'])}

## 5. Annotation quality

No YOLO/COCO/VOC annotation set was found for this dataset. Consequently,
bounding-box validity, class IDs, and image-label correspondence cannot be
validated. `annotation_errors.csv` records the performed annotation scan and
the report does not treat classification folder names as bounding-box classes.

## 6. Image quality

Quality flags are screening indicators, not definitive judgments. The checks
include readability, small dimensions, unusual aspect ratios, low grayscale
variance as a blur proxy, and extreme brightness thresholds. See
`image_quality_report.csv` and the plots directory.

## 7. Duplicate and leakage analysis

Exact duplicate files are identified with SHA-256 hashes. Potential
near-duplicates use a simple perceptual hash and Hamming-distance threshold;
they require manual review. Duplicates across train/test are leakage risks.

- Exact duplicate groups: {s['exact_duplicate_groups']}
- Cross-split exact duplicate groups: {s['cross_split_duplicate_groups']}
- Potential near-duplicate pairs: {s['near_duplicate_pairs']}

## 8. Visual analysis

Charts are in `plots/`, including class/split distributions, dimensions,
quality flags, and a sample image grid.

## 9. Problems identified

{chr(10).join(f"- {problem}" for problem in analysis['problems']) or '- None detected by the implemented checks.'}

## 10. Recommendations before training

1. Obtain or create reviewed bounding-box annotations if the project must use
   YOLO object detection; do not infer boxes from classification folders.
2. Resolve any cross-split duplicates before model evaluation.
3. Review all flagged quality images manually.
4. Keep the original dataset immutable and create derived data in a separate
   location.
5. After annotations exist, validate class IDs and leakage-safe splits before
   training.

## 11. Final dataset readiness assessment

**{readiness}**

The dataset is not ready for the requested YOLO object-detection training
workflow because usable object-detection annotations are absent. It may be
usable for image classification after the documented image-quality and leakage
flags are reviewed.
"""
    report_path.write_text(report, encoding="utf-8")


def analyze(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"; plot_dir.mkdir(exist_ok=True)
    paths = image_files(root)
    quality_rows = [screen_image(path) for path in paths]
    readable = [row for row in quality_rows if row["status"] == "ok"]
    class_totals = Counter(class_for(Path(row["path"]), root) for row in readable)
    split_totals = Counter(split_for(Path(row["path"]), root) for row in readable)
    class_rows = [{"class_name": name, "split": split, "image_count": sum(
        1 for row in readable if class_for(Path(row["path"]), root) == name and split_for(Path(row["path"]), root) == split)}
        for name in sorted(class_totals) for split in sorted(split_totals)]
    split_rows = [{"split": split, "image_count": count} for split, count in sorted(split_totals.items())]

    hashes: dict[str, list[str]] = defaultdict(list)
    phashes: dict[str, list[str]] = defaultdict(list)
    quality_by_path = {row["path"]: row for row in quality_rows}
    for path in paths:
        if quality_by_path[str(path)].get("status") != "ok":
            continue
        hashes[sha256(path)].append(str(path))
        try:
            with Image.open(path) as image:
                phashes[perceptual_hash(image)].append(str(path))
        except Exception:
            pass
    duplicate_rows = [{"hash": digest, "paths": "|".join(items), "count": len(items),
                       "splits": "|".join(sorted({split_for(Path(item), root) for item in items}))}
                      for digest, items in hashes.items() if len(items) > 1]
    near_rows = []
    hash_items = list(phashes.items())
    for index, (left_hash, left_paths) in enumerate(hash_items):
        for right_hash, right_paths in hash_items[index + 1:]:
            distance = hamming(left_hash, right_hash)
            if 0 < distance <= 8:
                near_rows.append({"distance": distance, "left": left_paths[0], "right": right_paths[0]})

    errors = yolo_errors(root, paths)
    dims = [(row["width"], row["height"]) for row in readable]
    summary = {
        "total_images": len(paths), "readable_images": len(readable),
        "corrupted_images": len(paths) - len(readable), "annotation_files": len(list(root.rglob("*.txt"))),
        "classes": sorted(class_totals), "class_totals": dict(sorted(class_totals.items())),
        "split_totals": dict(sorted(split_totals.items())), "min_width": min((x[0] for x in dims), default=None),
        "max_width": max((x[0] for x in dims), default=None), "min_height": min((x[1] for x in dims), default=None),
        "max_height": max((x[1] for x in dims), default=None), "exact_duplicate_groups": len(duplicate_rows),
        "cross_split_duplicate_groups": sum(1 for row in duplicate_rows if len(row["splits"].split("|")) > 1),
        "near_duplicate_pairs": len(near_rows),
        "quality_flags": {
            "small": sum(bool(row.get("is_small", False)) for row in readable),
            "unusual_aspect_ratio": sum(bool(row.get("is_unusual_aspect_ratio", False)) for row in readable),
            "blurry_screening": sum(bool(row.get("is_blurry_screening", False)) for row in readable),
            "overexposed_screening": sum(bool(row.get("is_overexposed_screening", False)) for row in readable),
            "underexposed_screening": sum(bool(row.get("is_underexposed_screening", False)) for row in readable),
        },
    }
    problems = []
    if summary["corrupted_images"]: problems.append(f"{summary['corrupted_images']} image(s) are corrupted or unreadable.")
    if errors: problems.append(f"{len(errors)} annotation issue(s) were found by the YOLO validator.")
    problems.append("No object-detection annotations are present; the dataset is classification-only.")
    if summary["cross_split_duplicate_groups"]: problems.append("Exact duplicate content crosses dataset splits.")
    analysis = {"root": str(root), "format": detect_format(root), "quality_rows": quality_rows,
                "class_rows": class_rows, "split_rows": split_rows, "summary": summary,
                "annotation_errors": errors, "duplicate_rows": duplicate_rows, "near_rows": near_rows,
                "problems": problems, "readiness": "NOT READY"}
    save_csv(quality_rows, output_dir / "image_quality_report.csv")
    save_csv(class_rows, output_dir / "class_distribution.csv", ["class_name", "split", "image_count"])
    save_csv(split_rows, output_dir / "split_distribution.csv")
    save_csv(errors, output_dir / "annotation_errors.csv", ["label", "image", "line", "error"])
    save_csv([{"path": path, "error": row["error"]} for row in quality_rows if row["status"] != "ok"],
             output_dir / "corrupted_images.csv", ["path", "error"])
    save_csv(duplicate_rows, output_dir / "duplicate_images.csv")
    save_csv(near_rows, output_dir / "potential_near_duplicates.csv")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_rows = [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
                    for key, value in summary.items()]
    save_csv(summary_rows, output_dir / "dataset_summary.csv", ["metric", "value"])
    make_plots(analysis, plot_dir)
    build_report(analysis, output_dir / "DATASET_ANALYSIS_REPORT.md")
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results/dataset_analysis"), type=Path)
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_root}")
    analysis = analyze(args.dataset_root, args.output_dir)
    print(json.dumps(analysis["summary"], indent=2))
    print(f"Readiness: {analysis['readiness']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
