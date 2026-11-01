"""Audit exact-image groups that carry conflicting source class labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def audit(duplicate_report: Path, manifest: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with duplicate_report.open(encoding="utf-8", newline="") as stream:
        groups = list(csv.DictReader(stream))
    with manifest.open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    conflicts = [row for row in groups if len(row["classes"].split("|")) > 1]
    audit_rows = []
    unresolved_rows = []
    quarantine_rows = []
    for group in conflicts:
        members = group["paths"].split("|")
        for source_path in members:
            path = Path(source_path)
            member = next(row for row in manifest_rows if row["source_path"] == source_path)
            actual_hash = sha256(path)
            audit_rows.append({
                "group_id": group["group_id"], "sha256_reported": group["sha256"],
                "sha256_verified": actual_hash, "hash_matches": actual_hash == group["sha256"],
                "source_path": source_path, "source_class": member["class_name"],
                "derived_split": member["split"], "derived_path": member["derived_path"],
                "conflicting_labels": group["classes"], "resolution_status": "unresolved",
                "resolution_reason": "No reliable metadata or documented human review establishes the correct label.",
            })
        unresolved_rows.append({
            "group_id": group["group_id"], "sha256": group["sha256"],
            "member_count": group["member_count"], "conflicting_labels": group["classes"],
            "source_splits": group["source_splits"], "paths": group["paths"],
            "resolution_status": "unresolved", "resolution_reason": "Requires human label review.",
        })
        for member in manifest_rows:
            if member["exact_group_id"] == group["group_id"]:
                quarantine_rows.append({
                    "group_id": group["group_id"], "sha256": group["sha256"],
                    "source_path": member["source_path"], "source_class": member["class_name"],
                    "previous_derived_path": member["derived_path"], "previous_split": member["split"],
                    "quarantine_reason": "Unresolved exact-hash label conflict",
                })
    write_csv(audit_rows, output_dir / "label_conflict_audit.csv")
    write_csv([], output_dir / "verified_groups.csv")
    write_csv(unresolved_rows, output_dir / "unresolved_groups.csv")
    write_csv(quarantine_rows, output_dir / "quarantine_manifest.csv")
    summary = {
        "conflicting_groups": len(conflicts),
        "conflicting_images": len(quarantine_rows),
        "verified_groups": 0,
        "unresolved_groups": len(unresolved_rows),
        "hash_verification_failures": sum(row["hash_matches"] is False for row in audit_rows),
    }
    report = f"""# Phase 3.1 Label Conflict Resolution Report

## Result

- Conflicting exact-hash groups audited: **{summary['conflicting_groups']}**
- Conflicting images audited: **{summary['conflicting_images']}**
- Verified groups: **0**
- Unresolved groups: **{summary['unresolved_groups']}**
- SHA-256 verification failures: **{summary['hash_verification_failures']}**

Every audited member was re-hashed from the source path. The conflicting
groups contain identical file content but different source folder labels.
There is no reliable source metadata or documented human review that can
establish which label is correct. No majority-label rule was applied.

All unresolved groups are quarantined from the rebuilt derived dataset. The
original source files remain untouched. Review `unresolved_groups.csv` and
`quarantine_manifest.csv` to resolve these groups manually in the future.
"""
    (output_dir / "conflict_resolution_report.md").write_text(report, encoding="utf-8")
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplicate-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results/phase3_1"), type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.duplicate_report, args.manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
