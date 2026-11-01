import pytest

from app.utils.explanation import build_explanation, confidence_band, sorted_probabilities


PROBABILITIES = {
    "Blotch_Apple": 0.12,
    "Normal_Apple": 0.03,
    "Rot_Apple": 0.80,
    "Scab_Apple": 0.05,
}


def test_probability_sorting_preserves_all_classes() -> None:
    ranked = sorted_probabilities(PROBABILITIES)
    assert [name for name, _ in ranked] == [
        "Rot_Apple", "Blotch_Apple", "Scab_Apple", "Normal_Apple"
    ]
    assert sum(value for _, value in ranked) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("confidence", "label"),
    [(0.85, "High model confidence"), (0.6499, "Low model confidence"), (0.65, "Moderate model confidence")],
)
def test_confidence_bands_are_presentation_labels(confidence: float, label: str) -> None:
    assert confidence_band(confidence)[0] == label


def test_explanation_uses_model_outputs_without_visual_claims() -> None:
    explanation = build_explanation("Rot_Apple", PROBABILITIES)
    assert "Apple Rot" in explanation
    assert "80.00%" in explanation
    assert "Apple Blotch (12.00%)" in explanation
    assert "does not identify a specific visual cause" in explanation


def test_explanation_rejects_non_top_class() -> None:
    with pytest.raises(ValueError):
        build_explanation("Scab_Apple", PROBABILITIES)


def test_probability_sorting_rejects_missing_class() -> None:
    with pytest.raises(ValueError):
        sorted_probabilities({"Rot_Apple": 1.0})
