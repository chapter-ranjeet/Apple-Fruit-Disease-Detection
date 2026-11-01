from pathlib import Path

import pytest
import torch
from PIL import Image

from app.utils.history import append_prediction, read_history
from app.utils.inference import CLASS_NAMES, resolve_device
from app.utils.validation import validate_image_bytes


def test_image_validation_accepts_png_and_rejects_bad_extension() -> None:
    image = Image.new("RGB", (4, 4), "green")
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    assert validate_image_bytes(buffer.getvalue(), "apple.png").size == (4, 4)
    with pytest.raises(ValueError):
        validate_image_bytes(buffer.getvalue(), "apple.gif")


def test_corrupt_image_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_image_bytes(b"not an image", "apple.png")


def test_cpu_fallback_and_class_mapping() -> None:
    expected = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert resolve_device("cuda:0") == expected
    assert resolve_device("cuda:999") == "cpu"
    assert CLASS_NAMES == ("Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple")


def test_prediction_history_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    record = {"filename": "apple.png", "predicted_class": "Rot_Apple",
              "confidence": 0.9, "device": "cpu"}
    assert append_prediction(path, record, "same") is True
    assert append_prediction(path, record, "same") is False
    assert len(read_history(path)) == 1


def test_missing_checkpoint_is_explicit() -> None:
    from app.utils.inference import load_model
    with pytest.raises(FileNotFoundError):
        load_model(Path("missing-best.pt"))
