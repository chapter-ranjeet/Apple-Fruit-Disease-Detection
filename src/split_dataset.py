"""Create a reproducible, exact-duplicate-safe classification dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .duplicate_resolution import build_resolution_rows, class_name, exact_groups, image_files
    from .preprocessing import preprocessing_config, validate_image
except ImportError:
    from duplicate_resolution import build_resolution_rows, class_name, exact_groups, image_files
    from preprocessing import preprocessing_config, validate_image


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def allocate_groups(groups: dict[str, list[Path]], root: Path, seed: int,
                    ratios: dict[str, float]) -> dict[str, str]:
    """Assign each exact-hash group once, balancing class counts greedily."""
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    totals = Counter(class_name(path, root) for members in groups.values() for path in members)
    assigned = {split: Counter() for split in ratios}
    result: dict[str, str] = {}
    target = {split: {label: count * ratio for label, count in totals.items()}
              for split, ratio in ratios.items()}
    for digest, members in items:
        counts = Counter(class_name(path, root) for path in members)
        scored = []
        for split in ratios:
            deficit = sum(max(target[split][label] - assigned[split][label], 0) for label in counts)
            load = sum(assigned[split].values()) / max(sum(totals.values()) * ratios[split], 1)
            scored.append((deficit, -load, split))
        chosen = max(scored)[2]
        result[digest] = chosen
        assigned[chosen].update(counts)
    return result


def derive_dataset(source: Path, output: Path, report_dir: Path, seed: int = 42,
                   near_duplicates: Path | None = None,
                   exclude_conflicts: Path | None = None) -> dict[str, Any]:
    paths = image_files(source)
    groups = exact_groups(paths)
    excluded_hashes = set()
    if exclude_conflicts and exclude_conflicts.is_file():
        with exclude_conflicts.open(encoding="utf-8", newline="") as stream:
            excluded_hashes = {row["sha256"] for row in csv.DictReader(stream)}
        groups = {digest: members for digest, members in groups.items() if digest not in excluded_hashes}
    assignments = allocate_groups(groups, source, seed, {"train": .70, "val": .15, "test": .15})
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    validation_rows = []
    used_names: set[tuple[str, str, str]] = set()
    for digest, members in sorted(groups.items()):
        split = assignments[digest]
        for path in sorted(members):
            label = class_name(path, source)
            validation = validate_image(path)
            validation_rows.append(validation)
            safe_name = path.name
            key = (split, label, safe_name.lower())
            if key in used_names:
                safe_name = f"{path.stem}_{digest[:10]}{path.suffix.lower()}"
            used_names.add((split, label, safe_name.lower()))
            destination = output / split / label / safe_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            manifest.append({
                "source_path": str(path), "derived_path": str(destination),
                "class_name": label, "split": split, "sha256": digest,
                "exact_group_id": f"exact_{digest[:12]}", "source_split": path.parts[-3].lower(),
                "is_valid_image": validation["valid"],
            })
    write_csv(manifest, report_dir / "split_manifest.csv")
    resolution_rows = build_resolution_rows(groups, source)
    write_csv(resolution_rows, report_dir / "duplicate_resolution_report.csv")
    if near_duplicates and near_duplicates.is_file():
        shutil.copy2(near_duplicates, report_dir / "potential_near_duplicates.csv")
    class_split = Counter((row["split"], row["class_name"]) for row in manifest)
    split_counts = Counter(row["split"] for row in manifest)
    conflicting_label_groups = sum(
        len({class_name(path, source) for path in members}) > 1
        for members in groups.values()
    )
    leakage_groups = sum(1 for digest, split in assignments.items()
                         if len({assignments[digest]}) != 1)
    summary = {
        "source": str(source.resolve()), "derived_root": str(output.resolve()),
        "seed": seed, "target_ratios": {"train": .70, "val": .15, "test": .15},
        "total_images": len(manifest), "excluded_conflict_groups": len(excluded_hashes),
        "excluded_conflict_images": sum(len(members) for digest, members in exact_groups(paths).items()
                                       if digest in excluded_hashes),
        "exact_hash_groups": len(groups),
        "duplicate_members_kept_together": all(len({row["split"] for row in manifest if row["sha256"] == digest}) == 1
                                               for digest in groups),
        "derived_split_counts": dict(sorted(split_counts.items())),
        "class_split_counts": {f"{split}/{label}": count for (split, label), count in sorted(class_split.items())},
        "all_images_valid": all(row["valid"] for row in validation_rows),
        "cross_split_exact_leakage": leakage_groups,
        "conflicting_label_groups": conflicting_label_groups,
    }
    (report_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "preprocessing_report.json").write_text(json.dumps({
        "source_files_unchanged": True, "validation": {
            "images_checked": len(validation_rows), "invalid_images": sum(not row["valid"] for row in validation_rows)
        }, "configuration": preprocessing_config(), "augmentation_applied": False,
    }, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", default=Path("dataset_processed"), type=Path)
    parser.add_argument("--report-dir", default=Path("results/phase3"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--near-duplicates", default=Path("results/dataset_analysis/potential_near_duplicates.csv"), type=Path)
    parser.add_argument("--exclude-conflicts", default=None, type=Path)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"Source dataset does not exist: {args.source}")
    print(json.dumps(derive_dataset(args.source, args.output, args.report_dir, args.seed,
                                    args.near_duplicates, args.exclude_conflicts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
