"""Exact duplicate grouping and leakage-safe split planning."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    from .preprocessing import IMAGE_EXTENSIONS
except ImportError:
    from preprocessing import IMAGE_EXTENSIONS


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() in {"train", "val", "validation", "test"} and index + 1 < len(parts) - 1:
            return parts[index + 1]
    return parts[-2] if len(parts) > 1 else "unclassified"


def source_split(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if part.lower() in {"train", "val", "validation", "test"}:
            return "val" if part.lower() == "validation" else part.lower()
    return "unsplit"


def exact_groups(paths: Iterable[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[sha256(path)].append(path)
    return dict(groups)


def build_resolution_rows(groups: dict[str, list[Path]], root: Path) -> list[dict[str, object]]:
    rows = []
    for digest, members in sorted(groups.items()):
        splits = sorted({source_split(path, root) for path in members})
        rows.append({
            "sha256": digest,
            "group_id": f"exact_{digest[:12]}",
            "member_count": len(members),
            "source_splits": "|".join(splits),
            "cross_split": len(splits) > 1,
            "classes": "|".join(sorted({class_name(path, root) for path in members})),
            "paths": "|".join(str(path) for path in sorted(members)),
            "resolution": "keep_one_group_in_one_derived_split",
        })
    return rows
