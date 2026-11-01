"""Run the image quality screening independently."""

from pathlib import Path
import argparse
import pandas as pd

from dataset_analysis import image_files, screen_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("results/dataset_analysis/image_quality_report.csv"), type=Path)
    args = parser.parse_args()
    rows = [screen_image(path) for path in image_files(args.dataset_root)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Screened {len(rows)} images; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
