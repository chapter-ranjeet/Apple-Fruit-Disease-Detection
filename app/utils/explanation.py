from __future__ import annotations

from typing import Mapping

from .inference import CLASS_NAMES, DISPLAY_NAMES


def confidence_band(confidence: float) -> tuple[str, str]:
    """Return a presentation-only confidence label and guidance."""
    if confidence >= 0.85:
        return "High model confidence", "The model strongly favored this category among the supported classes."
    if confidence >= 0.65:
        return "Moderate model confidence", "The model favored this category, but the alternatives remain relevant."
    return "Low model confidence", "The model did not strongly separate the supported categories."


def sorted_probabilities(probabilities: Mapping[str, float]) -> list[tuple[str, float]]:
    """Sort the complete model probability vector without changing its values."""
    missing = set(CLASS_NAMES) - set(probabilities)
    if missing:
        raise ValueError(f"Missing probability classes: {sorted(missing)}")
    return sorted(
        ((class_name, float(probabilities[class_name])) for class_name in CLASS_NAMES),
        key=lambda item: item[1],
        reverse=True,
    )


def build_explanation(predicted_class: str, probabilities: Mapping[str, float]) -> str:
    """Explain only what can be supported by the model output."""
    if predicted_class not in CLASS_NAMES:
        raise ValueError("Unknown predicted class.")
    ranked = sorted_probabilities(probabilities)
    highest_name, highest_value = ranked[0]
    if highest_name != predicted_class:
        raise ValueError("Predicted class must have the highest probability.")
    alternatives = ", ".join(
        f"{DISPLAY_NAMES[name]} ({value:.2%})" for name, value in ranked[1:]
    )
    return (
        f"The model assigned the highest probability to {DISPLAY_NAMES[predicted_class]} "
        f"({highest_value:.2%}) among the four supported categories. "
        f"The other model outputs were {alternatives}. This describes the model's "
        "relative classification preference; it does not identify a specific visual "
        "cause and is not proof of disease or diagnostic certainty."
    )
