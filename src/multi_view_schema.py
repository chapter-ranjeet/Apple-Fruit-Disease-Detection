"""Schema and future fusion design for reviewed four-view apple records."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

VIEW_NAMES = ("front", "back", "upper", "lower")


@dataclass(frozen=True)
class MultiViewRecord:
    apple_id: str
    front_image: str | None
    back_image: str | None
    upper_image: str | None
    lower_image: str | None
    disease_label: str | None = None


def validate_record(record: MultiViewRecord, require_all_views: bool = True) -> list[str]:
    errors = []
    if not record.apple_id.strip():
        errors.append("apple_id is required")
    views = {name: getattr(record, f"{name}_image") for name in VIEW_NAMES}
    if require_all_views and any(not value for value in views.values()):
        errors.append("all four reviewed views are required")
    if record.disease_label is not None and not record.disease_label.strip():
        errors.append("disease_label must be non-empty when supplied")
    return errors


def future_fusion_design() -> dict[str, Any]:
    return {
        "input_schema": ["front_image", "back_image", "upper_image", "lower_image"],
        "per_view_output": "probability vector over reviewed disease classes",
        "aggregation": "mean probability across available views, with view count recorded",
        "supporting_views": "views whose class probability meets the configured support threshold",
        "disagreement": "report max pairwise probability-vector distance and top-class disagreement",
        "missing_views": "allow only when explicitly configured; report missing views and lower confidence",
        "ground_truth_requirement": "apple-level reviewed four-view records are required for fusion evaluation",
        "unsupported_claim": "classification probabilities do not localize disease symptoms",
    }


def record_to_dict(record: MultiViewRecord) -> dict[str, Any]:
    return asdict(record)
