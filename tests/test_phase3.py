from pathlib import Path

from PIL import Image

from src.duplicate_resolution import exact_groups
from src.multi_view_schema import MultiViewRecord, validate_record
from src.preprocessing import validate_image
from src.split_dataset import allocate_groups


def _image(path: Path, color: str = "red") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color).save(path)
    return path


def test_exact_duplicates_are_grouped(tmp_path: Path) -> None:
    first = _image(tmp_path / "Train" / "A" / "one.png")
    second = _image(tmp_path / "Test" / "A" / "two.png")
    assert len(exact_groups([first, second])) == 1


def test_group_assignment_prevents_leakage(tmp_path: Path) -> None:
    first = _image(tmp_path / "Train" / "A" / "one.png")
    second = _image(tmp_path / "Test" / "A" / "two.png")
    groups = exact_groups([first, second])
    assignment = allocate_groups(groups, tmp_path, 42, {"train": .7, "val": .15, "test": .15})
    assert len(set(assignment.values())) == 1


def test_image_validation(tmp_path: Path) -> None:
    result = validate_image(_image(tmp_path / "valid.png"))
    assert result["valid"] is True


def test_multiview_record_validation() -> None:
    record = MultiViewRecord("apple-1", "f.png", "b.png", "u.png", "l.png", "Normal_Apple")
    assert validate_record(record) == []
    assert validate_record(MultiViewRecord("apple-2", "f.png", None, None, None)) == [
        "all four reviewed views are required"
    ]
