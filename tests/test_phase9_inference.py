from pathlib import Path

import pytest

from app.utils.inference import CLASS_NAMES, load_model


def test_missing_checkpoint_does_not_expose_or_load() -> None:
    with pytest.raises(FileNotFoundError):
        load_model(Path("does-not-exist-best.pt"))


def test_expected_class_mapping() -> None:
    assert CLASS_NAMES == ("Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple")
