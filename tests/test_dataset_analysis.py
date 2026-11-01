from pathlib import Path

from PIL import Image

from src.dataset_analysis import detect_format, perceptual_hash, yolo_errors


def test_detects_folder_classification(tmp_path: Path) -> None:
    (tmp_path / "Train" / "Healthy").mkdir(parents=True)
    assert detect_format(tmp_path) == "Folder-based image classification"


def test_validates_yolo_rows(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), "white").save(image)
    (tmp_path / "image.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert yolo_errors(tmp_path, [image]) == []


def test_rejects_bad_yolo_rows(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), "white").save(image)
    (tmp_path / "image.txt").write_text("0 2 0.5 0 0.2\n", encoding="utf-8")
    assert len(yolo_errors(tmp_path, [image])) >= 2


def test_perceptual_hash_is_stable(tmp_path: Path) -> None:
    image = Image.new("RGB", (10, 10), "white")
    assert perceptual_hash(image) == perceptual_hash(image)
