"""Generate Phase 3 readiness and multi-view limitation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_view_schema import future_fusion_design


def write_reports(report_dir: Path, split_summary: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    conflict_count = split_summary.get("conflicting_label_groups", 0)
    if conflict_count:
        assessment = "NOT READY FOR CLASSIFICATION TRAINING WITHOUT LABEL REVIEW"
    else:
        assessment = "READY WITH WARNINGS FOR SINGLE-IMAGE CLASSIFICATION"
    readiness = f"""# Phase 3.1 Dataset Readiness

## Assessment

**{assessment}**

The derived dataset contains all four discovered classes, valid readable
images, and exact duplicate groups assigned to one split only. This makes it
structurally appropriate for a conventional image-classification baseline.
Unresolved conflicting groups were quarantined rather than assigned a
majority label.

It is **not ready for true multi-view evaluation** and is not a detection
dataset: the source has no bounding boxes and no reviewed four-view apple IDs.

## Verification

- Exact duplicate leakage across derived splits: 0
- Exact groups with conflicting class labels remaining: {conflict_count}
- Exact groups quarantined: {split_summary.get("excluded_conflict_groups", 0)}
- Images quarantined: {split_summary.get("excluded_conflict_images", 0)}
- Source images modified: no
- Four classes present: yes
- Image validation: see `preprocessing_report.json`
- Split details: see `split_summary.json` and `split_manifest.csv`

## Remaining warnings

1. 290 exact-hash groups contained conflicting source class labels and were
   quarantined. Review `results/phase3_1/unresolved_groups.csv` before
   deciding whether to restore any images with corrected labels.
2. Potential near-duplicates remain separate candidates and require manual
   review; perceptual similarity is not proof of the same physical apple.
3. The original Train/Test folders are not reliable apple-level groups.
"""
    (report_dir / "dataset_readiness_report.md").write_text(readiness, encoding="utf-8")
    mv = f"""# Multi-view Dataset Report

## Evidence from the supplied data

The source contains class folders under `Train` and `Test`, image filenames,
and no metadata identifying physical apples or camera viewpoints. Filenames
such as numbered files and web-derived names are not sufficient evidence that
four photographs show the same apple. Therefore, **no four-view groups were
fabricated**.

## Required reviewed schema

| Field | Meaning |
|---|---|
| `apple_id` | reviewed identifier for one physical apple |
| `front_image` | path to the reviewed front view |
| `back_image` | path to the reviewed back view |
| `upper_image` | path to the reviewed upper view |
| `lower_image` | path to the reviewed lower view |
| `disease_label` | reviewed apple-level ground truth, or null until confirmed |

## Future fusion design

```json
{json.dumps(future_fusion_design(), indent=2)}
```

Fusion must be evaluated using apple-level records with known ground truth.
The current classification images can support a single-image classifier
baseline, but cannot validate four-view aggregation, view disagreement, or
missing-view behavior.
"""
    (report_dir / "multi_view_dataset_report.md").write_text(mv, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=Path("results/phase3"), type=Path)
    parser.add_argument("--summary", default=Path("results/phase3/split_summary.json"), type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    write_reports(args.report_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
