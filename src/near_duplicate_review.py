"""Generate and resolve human review records for perceptual near-duplicates.

Generation is fully automatic for evidence only. Resolution requires explicit
reviewer decisions and never infers a decision from similarity or labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

DECISIONS = {"same_image", "same_apple_different_view", "different_apple", "uncertain"}
REQUIRED_COLUMNS = [
    "pair_id", "image_a", "image_b", "class_a", "class_b", "split_a", "split_b",
    "similarity_score", "reviewer_decision", "reviewer_notes", "reviewed_by", "reviewed_at",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] = REQUIRED_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def cross_split_pairs(near_path: Path, manifest_path: Path) -> list[dict[str, str]]:
    near = read_csv(near_path)
    manifest = {row["source_path"]: row for row in read_csv(manifest_path)}
    pairs = []
    for index, row in enumerate(near, 1):
        left, right = manifest.get(row["left"]), manifest.get(row["right"])
        if left is None or right is None:
            raise ValueError(f"Near-duplicate path missing from manifest: {row['left']} / {row['right']}")
        if left["split"] == right["split"]:
            continue
        pairs.append({
            "pair_id": f"ND-{len(pairs) + 1:03d}",
            "image_a": row["left"], "image_b": row["right"],
            "class_a": left["class_name"], "class_b": right["class_name"],
            "split_a": left["split"], "split_b": right["split"],
            "similarity_score": row["distance"],
            "reviewer_decision": "", "reviewer_notes": "",
            "reviewed_by": "", "reviewed_at": "",
        })
    return pairs


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(row: dict[str, str], output: Path) -> None:
    canvas = Image.new("RGB", (1800, 1150), "white")
    draw = ImageDraw.Draw(canvas)
    title = _font(32)
    body = _font(22)
    draw.text((40, 25), f"{row['pair_id']} | perceptual distance: {row['similarity_score']}", fill="black", font=title)
    for offset, key in ((40, "image_a"), (920, "image_b")):
        path = Path(row[key])
        try:
            with Image.open(path) as source:
                preview = ImageOps.contain(source.convert("RGB"), (820, 820))
            x = offset + (820 - preview.width) // 2
            canvas.paste(preview, (x, 95))
            draw.text((offset, 930), f"{key}: {path.name}", fill="black", font=body)
            draw.text((offset, 970), f"class={row['class_a' if key == 'image_a' else 'class_b']} | "
                                     f"split={row['split_a' if key == 'image_a' else 'split_b']}",
                      fill="black", font=body)
            draw.text((offset, 1010), str(path), fill="black", font=_font(16))
        except Exception as exc:
            draw.text((offset, 500), f"Unable to load image: {exc}", fill="red", font=body)
    canvas.save(output)


def generate_review(near_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, int]:
    pairs = cross_split_pairs(near_path, manifest_path)
    sheets = output_dir / "review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for row in pairs:
        make_sheet(row, sheets / f"{row['pair_id']}.png")
    write_csv(pairs, output_dir / "near_duplicate_human_review.csv")
    (output_dir / "final_manifest.csv").write_text(
        "source_path,derived_path,class_name,split,sha256,review_status\n", encoding="utf-8"
    )
    (output_dir / "final_dataset_summary.json").write_text(json.dumps({
        "status": "pending_human_review", "candidate_pairs": len(pairs),
        "reviewed_pairs": 0, "approved_images": 0,
    }, indent=2), encoding="utf-8")
    return {"candidate_pairs": len(pairs), "review_sheets": len(list(sheets.glob("*.png")))}


def resolve_review(review_path: Path, source_manifest: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_csv(review_path)
    missing = [row["pair_id"] for row in rows if row["reviewer_decision"] not in DECISIONS]
    if missing:
        raise ValueError(f"Human review is incomplete or invalid for pair(s): {', '.join(missing)}")
    manifest = {row["source_path"]: row for row in read_csv(source_manifest)}
    excluded: dict[str, str] = {}
    for row in rows:
        if row["reviewer_decision"] in {"same_image", "same_apple_different_view", "uncertain"}:
            for path in (row["image_a"], row["image_b"]):
                excluded[path] = row["reviewer_decision"]
    approved = [row for path, row in manifest.items() if path not in excluded]
    write_csv(approved, output_dir / "final_manifest.csv",
              ["source_path", "derived_path", "class_name", "split", "sha256", "is_valid_image"])
    write_csv([{"source_path": path, "reason": reason} for path, reason in sorted(excluded.items())],
              output_dir / "excluded_images.csv", ["source_path", "reason"])
    summary = {
        "status": "resolved", "reviewed_pairs": len(rows),
        "approved_images": len(approved), "excluded_images": len(excluded),
        "decisions": {decision: sum(row["reviewer_decision"] == decision for row in rows)
                      for decision in sorted(DECISIONS)},
        "note": "Rebuild must preserve group-aware split and be performed from approved records.",
    }
    (output_dir / "final_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--near-duplicates", required=True, type=Path)
    generate.add_argument("--manifest", required=True, type=Path)
    generate.add_argument("--output-dir", default=Path("results/phase4_1"), type=Path)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--review", required=True, type=Path)
    resolve.add_argument("--manifest", required=True, type=Path)
    resolve.add_argument("--output-dir", default=Path("results/phase4_1"), type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate_review(args.near_duplicates, args.manifest, args.output_dir), indent=2))
    else:
        print(json.dumps(resolve_review(args.review, args.manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
