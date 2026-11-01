from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image

from app.utils.history import append_prediction, read_history
from app.utils.inference import resolve_device
from app.utils.validation import safe_filename, validate_image_bytes


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "green").save(buffer, format="PNG")
    return buffer.getvalue()


def test_filename_is_basename_and_control_safe() -> None:
    assert safe_filename("../../secret.png") == "secret.png"
    assert "\n" not in safe_filename("bad\nname.png")


def test_formula_safe_history_and_duplicate_detection(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    record = {
        "filename": "=HYPERLINK('https://example.invalid')",
        "predicted_class": "Rot_Apple",
        "confidence": 0.9,
        "device": "cpu",
    }
    assert append_prediction(history, record, "signature") is True
    assert append_prediction(history, record, "signature") is False
    row = read_history(history)[0]
    assert not row["filename"].startswith(("=", "+", "-", "@"))
    assert "example.invalid" in row["filename"]
    assert len(read_history(history)) == 1


def test_image_size_and_extension_limits() -> None:
    with pytest.raises(ValueError):
        validate_image_bytes(png_bytes(), "apple.gif")
    with pytest.raises(ValueError):
        validate_image_bytes(b"", "apple.png")


def test_cpu_fallback_is_safe() -> None:
    expected = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert resolve_device("cuda:0") == expected
    assert resolve_device("cuda:999") == "cpu"
