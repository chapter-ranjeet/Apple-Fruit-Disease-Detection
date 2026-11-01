"""Validate YOLO annotations without changing source files."""

from pathlib import Path
import argparse
import sys

from dataset_analysis import image_files, yolo_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    errors = yolo_errors(args.dataset_root, image_files(args.dataset_root))
    print(f"YOLO annotation errors: {len(errors)}")
    for error in errors[:20]:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
