from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import re
import unicodedata

from PIL import Image, ImageFile, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def safe_filename(filename: str) -> str:
    name = unicodedata.normalize("NFKC", Path(filename).name)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    if not name or name in {".", ".."}:
        return "uploaded_image"
    return name[:120]


def validate_image_bytes(data: bytes, filename: str = "") -> Image.Image:
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The image exceeds the 10 MB upload limit.")
    extension = Path(safe_filename(filename)).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only JPG, JPEG, and PNG images are supported.")
    try:
        image = Image.open(BytesIO(data))
        image.verify()
        return Image.open(BytesIO(data)).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning,
            UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc


def read_uploaded_file(uploaded_file: BinaryIO) -> tuple[bytes, Image.Image]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = uploaded_file.read(min(1024 * 1024, MAX_UPLOAD_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise ValueError("The image exceeds the 10 MB upload limit.")
    data = b"".join(chunks)
    return data, validate_image_bytes(data, getattr(uploaded_file, "name", ""))
