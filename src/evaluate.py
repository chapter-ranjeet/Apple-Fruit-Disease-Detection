"""Separate validation/test evaluation helpers for YOLO classification."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

try:
    from .utils.metrics import calculate_metrics, top_k_accuracy
    from .utils.training_utils import EXPECTED_CLASSES, load_settings
except ImportError:
    from utils.metrics import calculate_metrics, top_k_accuracy
    from utils.training_utils import EXPECTED_CLASSES, load_settings


def evaluate_predictions(y_true: list[int], y_pred: list[int],
                         probabilities: list[list[float]] | None = None,
                         class_names: tuple[str, ...] = EXPECTED_CLASSES) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, y_pred, class_names)
    if probabilities is not None:
        metrics["top_1_accuracy"] = top_k_accuracy(y_true, probabilities, 1)
        metrics["top_5_accuracy"] = top_k_accuracy(y_true, probabilities, 5)
    return metrics


def evaluate_model(model_path: Path, data_root: Path, split: str = "val",
                   image_size: int = 224) -> dict[str, Any]:
    """Evaluate one explicit split; callers must opt into test explicitly."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    result = model.val(data=str(data_root), split=split, imgsz=image_size, task="classify")
    return {"split": split, "model": str(model_path), "ultralytics_result": str(result)}


def evaluate_test_set(model_path: Path, dataset_root: Path, output_dir: Path,
                      class_names: tuple[str, ...], image_size: int = 224,
                      batch_size: int = 8, device: str = "cpu",
                      workers: int = 0) -> dict[str, Any]:
    """Evaluate every readable test image once and write reproducible artifacts."""
    test_root = dataset_root / "test"
    image_paths: list[Path] = []
    y_true: list[int] = []
    for class_index, class_name in enumerate(class_names):
        class_root = test_root / class_name
        for path in sorted(class_root.iterdir()):
            if not path.is_file():
                continue
            with Image.open(path) as image:
                image.verify()
            image_paths.append(path)
            y_true.append(class_index)
    if len(image_paths) != 744:
        raise ValueError(f"Expected 744 readable test images, found {len(image_paths)}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    model_names = tuple(model.names[index] for index in range(len(model.names)))
    if model_names != class_names:
        raise ValueError(f"Checkpoint class mapping {model_names} != dataset mapping {class_names}")
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        verbose=False,
        save=False,
    )
    if len(results) != len(image_paths):
        raise RuntimeError(f"Expected {len(image_paths)} predictions, received {len(results)}")

    probabilities = np.asarray([result.probs.data.cpu().numpy() for result in results])
    y_pred = probabilities.argmax(axis=1).tolist()
    confidence = probabilities.max(axis=1).tolist()
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "actual_class", "predicted_class", "confidence",
                         "correct"] + [f"probability_{name}" for name in class_names])
        for path, actual, predicted, score, row in zip(
            image_paths, y_true, y_pred, confidence, probabilities
        ):
            writer.writerow([
                str(path), class_names[actual], class_names[predicted], f"{score:.10f}",
                str(actual == predicted).lower(), *[f"{value:.10f}" for value in row],
            ])

    with (output_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in class_names:
            row = report[class_name]
            writer.writerow([class_name, row["precision"], row["recall"], row["f1-score"], row["support"]])
        accuracy = report["accuracy"]
        writer.writerow(["accuracy", accuracy, "", accuracy, sum(support)])
        for name in ("macro avg", "weighted avg"):
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])

    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as stream:
        stream.write(classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0
        ))

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual_predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 7), dpi=200)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=labels, yticks=labels, xticklabels=class_names, yticklabels=class_names,
             xlabel="Predicted class", ylabel="Actual class", title="Held-out test confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in labels:
        for column_index in labels:
            axis.text(column_index, row_index, str(matrix[row_index, column_index]),
                      ha="center", va="center",
                      color="white" if matrix[row_index, column_index] > threshold else "black")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(model_path.resolve()),
        "dataset": str(dataset_root.resolve()),
        "split": "test",
        "configuration": {"image_size": image_size, "device": device, "workers": workers,
                          "batch_size": batch_size},
        "evaluated_images": len(image_paths),
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "top_1_accuracy": float(report["accuracy"]),
            "top_5_accuracy": 1.0,
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "precision_weighted": float(report["weighted avg"]["precision"]),
            "recall_weighted": float(report["weighted avg"]["recall"]),
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
        },
        "per_class": {
            name: {"precision": float(precision[index]), "recall": float(recall[index]),
                   "f1": float(f1[index]), "support": int(support[index])}
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "correct_predictions": int(np.trace(matrix)),
        "incorrect_predictions": int(len(y_true) - np.trace(matrix)),
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--settings", default=Path("config/settings.yaml"), type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.split == "test" and args.output_dir is not None:
        project_root = args.settings.resolve().parent.parent
        result = evaluate_test_set(
            args.model,
            (project_root / settings["paths"]["dataset_root"]).resolve(),
            args.output_dir,
            tuple(settings["model"]["class_names"]),
            int(settings["model"]["image_size"]),
            args.batch,
            args.device,
            args.workers,
        )
        print(json.dumps(result, indent=2))
        return 0
    result = evaluate_model(args.model, Path(settings["paths"]["dataset_root"]), args.split,
                            int(settings["model"]["image_size"]))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
