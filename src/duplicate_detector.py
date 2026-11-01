"""Run duplicate detection independently."""

from pathlib import Path
import argparse
from collections import defaultdict
import pandas as pd
from PIL import Image

from dataset_analysis import image_files, perceptual_hash, sha256, split_for, screen_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results/dataset_analysis"), type=Path)
    args = parser.parse_args()
    exact = defaultdict(list)
    perceptual = defaultdict(list)
    for path in image_files(args.dataset_root):
        if screen_image(path)["status"] != "ok":
            continue
        exact[sha256(path)].append(str(path))
        with Image.open(path) as image:
            perceptual[perceptual_hash(image)].append(str(path))
    duplicates = [{"hash": key, "paths": "|".join(value), "count": len(value),
                   "splits": "|".join(sorted({split_for(Path(item), args.dataset_root) for item in value}))}
                  for key, value in exact.items() if len(value) > 1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(duplicates).to_csv(args.output_dir / "duplicate_images.csv", index=False)
    print(f"Exact duplicate groups: {len(duplicates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
