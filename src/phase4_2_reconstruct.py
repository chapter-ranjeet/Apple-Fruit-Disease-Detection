"""Audit approved near-duplicate decisions and rebuild the final dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ALLOWED = {"same_image", "same_apple_different_view", "different_apple", "uncertain"}
EXPECTED = {"same_image": 9, "different_apple": 28, "uncertain": 2, "same_apple_different_view": 0}
CLASSES = {"Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_review(review_rows: list[dict[str, str]], near_rows: list[dict[str, str]],
                    manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    expected_ids = {f"ND-{index:03d}" for index in range(1, 40)}
    actual_ids = [row.get("pair_id", "") for row in review_rows]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Review CSV must contain each pair ID ND-001 through ND-039 exactly once.")
    if any(row.get("user_confirmation") != "approved" for row in review_rows):
        raise ValueError("Every review row must have user_confirmation=approved.")
    if any(row.get("reviewed_by") != "Ranjeet" or row.get("reviewed_at") != "2026-11-01"
           for row in review_rows):
        raise ValueError("Every row must be reviewed by Ranjeet on 2026-11-01.")
    counts = Counter(row.get("reviewer_decision", "") for row in review_rows)
    if set(counts) - ALLOWED or counts != Counter(EXPECTED):
        raise ValueError(f"Decision counts do not match approved tally: {dict(counts)}")
    if len(near_rows) != 92:
        raise ValueError(f"Expected 92 near-duplicate source rows, found {len(near_rows)}.")
    source_manifest = {row["source_path"]: row for row in manifest_rows}
    pairs = []
    for index, near in enumerate(near_rows, 1):
        left, right = source_manifest.get(near["left"]), source_manifest.get(near["right"])
        if left is None or right is None or left["split"] == right["split"]:
            continue
        pairs.append({
            "pair_id": f"ND-{len(pairs) + 1:03d}", "image_a": near["left"], "image_b": near["right"],
            "class_a": left["class_name"], "class_b": right["class_name"],
            "split_a": left["split"], "split_b": right["split"], "similarity_score": near["distance"],
        })
    if len(pairs) != 39:
        raise ValueError(f"Expected 39 cross-split pairs, found {len(pairs)}.")
    return pairs, dict(counts)


def assign_groups(rows: list[dict[str, str]], excluded: set[str], seed: int = 42) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["source_path"] not in excluded]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        groups[row["sha256"]].append(row)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    totals = Counter(row["class_name"] for row in eligible)
    target = {"train": .70, "val": .15, "test": .15}
    assigned = {split: Counter() for split in target}
    group_split: dict[str, str] = {}
    for digest, members in items:
        labels = Counter(row["class_name"] for row in members)
        scores = []
        for split, ratio in target.items():
            deficit = sum(max(totals[label] * ratio - assigned[split][label], 0) for label in labels)
            load = sum(assigned[split].values()) / max(len(eligible) * ratio, 1)
            scores.append((deficit, -load, split))
        chosen = max(scores)[2]
        group_split[digest] = chosen
        assigned[chosen].update(labels)
    return [{**row, "final_split": group_split[row["sha256"]]} for row in eligible]


def reconstruct(source_manifest: Path, review_path: Path, near_path: Path, output_dir: Path,
                dataset_dir: Path, seed: int = 42) -> dict[str, Any]:
    manifest = read_csv(source_manifest)
    review = read_csv(review_path)
    near = read_csv(near_path)
    pairs, decision_counts = validate_review(review, near, manifest)
    review_by_id = {row["pair_id"]: row for row in review}
    excluded: dict[str, str] = {}
    exclusion_rows = []
    for pair in pairs:
        decision = review_by_id[pair["pair_id"]]["reviewer_decision"]
        if decision != "different_apple":
            for path in (pair["image_a"], pair["image_b"]):
                excluded.setdefault(path, decision)
    source_by_path = {row["source_path"]: row for row in manifest}
    for path, reason in sorted(excluded.items()):
        row = source_by_path[path]
        exclusion_rows.append({**row, "exclusion_reason": reason, "pair_id": next(
            pair["pair_id"] for pair in pairs if path in (pair["image_a"], pair["image_b"]))})
    final_rows = assign_groups(manifest, set(excluded), seed)
    if dataset_dir.exists():
        for child in dataset_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    for row in final_rows:
        source = Path(row["source_path"])
        destination = dataset_dir / row["final_split"] / row["class_name"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = dataset_dir / row["final_split"] / row["class_name"] / (
                f"{source.stem}_{row['sha256'][:10]}{source.suffix.lower()}"
            )
        shutil.copy2(source, destination)
        row["derived_path"] = str(destination)
        row["split"] = row.pop("final_split")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pairs, output_dir / "pair_resolution_audit.csv")
    write_csv(exclusion_rows, output_dir / "exclusion_audit.csv")
    write_csv(final_rows, output_dir / "final_dataset_manifest.csv")
    split_counts = Counter(row["split"] for row in final_rows)
    class_counts = Counter((row["split"], row["class_name"]) for row in final_rows)
    summary = {
        "status": "ready_with_warnings",
        "original_retained_before_review": len(manifest),
        "unique_excluded_images": len(excluded),
        "retained_images": len(final_rows),
        "decision_counts": decision_counts,
        "split_counts": dict(sorted(split_counts.items())),
        "class_split_counts": {f"{split}/{label}": count for (split, label), count in sorted(class_counts.items())},
        "classes": sorted({row["class_name"] for row in final_rows}),
        "seed": seed,
    }
    (output_dir / "final_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_final(summary: dict[str, Any], manifest_path: Path, dataset_dir: Path,
                   output_dir: Path) -> dict[str, Any]:
    rows = read_csv(manifest_path)
    errors = []
    seen_paths = set()
    hashes: dict[str, set[str]] = defaultdict(set)
    actual_counts = Counter()
    for row in rows:
        path = Path(row["derived_path"])
        if not path.is_file():
            errors.append(f"missing:{path}")
            continue
        if row["source_path"] in seen_paths:
            errors.append(f"duplicate_source_path:{row['source_path']}")
        seen_paths.add(row["source_path"])
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"unreadable:{path}:{exc}")
        actual_hash = sha256(path)
        hashes[actual_hash].add(row["split"])
        actual_counts[(row["split"], row["class_name"])] += 1
    leakage = sum(len(splits) > 1 for splits in hashes.values())
    classes = sorted({row["class_name"] for row in rows})
    report = {
        "manifest_rows": len(rows), "actual_dataset_files": sum(1 for _ in dataset_dir.rglob("*") if _.is_file()),
        "missing_or_unreadable": len(errors), "duplicate_manifest_paths": len(rows) - len(seen_paths),
        "exact_hash_groups_crossing_splits": leakage, "classes": classes,
        "all_four_classes_present": set(classes) == CLASSES,
        "all_four_classes_in_each_split": all(
            all((split, label) in actual_counts for label in CLASSES) for split in ("train", "val", "test")
        ),
        "actual_class_split_counts": {f"{split}/{label}": count for (split, label), count in sorted(actual_counts.items())},
        "validation_passed": not errors and leakage == 0 and set(classes) == CLASSES,
        "errors": errors[:20],
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=Path("results/phase3_1/split_manifest.csv"), type=Path)
    parser.add_argument("--review", default=Path("results/phase4_1/near_duplicate_human_review.csv"), type=Path)
    parser.add_argument("--near-duplicates", default=Path("results/phase3/potential_near_duplicates.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/phase4_2"), type=Path)
    parser.add_argument("--dataset-dir", default=Path("dataset_final"), type=Path)
    args = parser.parse_args()
    summary = reconstruct(args.source_manifest, args.review, args.near_duplicates, args.output_dir, args.dataset_dir)
    validation = validate_final(summary, args.output_dir / "final_dataset_manifest.csv", args.dataset_dir, args.output_dir)
    if not validation["validation_passed"]:
        raise SystemExit(json.dumps(validation, indent=2))
    print(json.dumps({**summary, "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
