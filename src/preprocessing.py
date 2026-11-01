"""Read-only image validation and preprocessing configuration for classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_image(path: Path) -> dict[str, Any]:
    """Validate an image and return metadata without altering the source."""
    result: dict[str, Any] = {"path": str(path), "valid": False, "error": ""}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result.update({
                "valid": True,
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_rgb(path: Path, image_size: tuple[int, int] | None = None) -> Image.Image:
    """Load an image as RGB, optionally resizing with aspect-ratio-preserving letterbox."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if image_size is not None:
            return ImageOps.contain(rgb, image_size)
        return rgb.copy()


def preprocessing_config(image_size: int = 224) -> dict[str, Any]:
    """Return the planned preprocessing and training-only augmentation policy."""
    return {
        "input_size": [image_size, image_size],
        "color_mode": "RGB",
        "resize": "aspect-ratio-preserving letterbox/padding during loading",
        "normalization": "model-specific normalization applied by the training framework",
        "validation_test_augmentation": False,
        "training_augmentation": {
            "horizontal_flip": "enabled only if acquisition protocol confirms it is label-preserving",
            "rotation_degrees": 10,
            "brightness_contrast": "mild",
            "zoom_crop": "mild",
            "preserve_symptoms": True,
        },
    }
