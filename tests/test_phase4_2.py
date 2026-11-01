from collections import Counter

from src.phase4_2_reconstruct import ALLOWED, EXPECTED


def test_approved_decision_tally() -> None:
    assert set(ALLOWED) >= set(EXPECTED)
    assert sum(EXPECTED.values()) == 39


def test_exclusion_is_union() -> None:
    pairs = [("a", "b"), ("b", "c")]
    excluded = {path for decision, pair in zip(("same_image", "uncertain"), pairs)
                if decision != "different_apple" for path in pair}
    assert excluded == {"a", "b", "c"}
