from pathlib import Path

import pytest
import torch

from src.train import build_train_args, validate_dataset
from src.utils.metrics import calculate_metrics, top_k_accuracy
from src.utils.training_utils import load_settings


def test_configuration_and_dataset() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    assert settings["model"]["task"] == "classify"
    assert settings["model"]["baseline"] == "yolo11n-cls.pt"
    assert settings["model"]["image_size"] == 224
    result = validate_dataset(Path("dataset_final"))
    assert result["valid"] is True
    assert result["total"] == 4964


def test_training_arguments_are_device_safe() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    args = build_train_args(settings, Path("runs/appleguard/classification"))
    assert args["task"] == "classify"
    assert args["device"] in {"cpu", "cuda:0"}
    assert args["device"] == ("cuda:0" if torch.cuda.is_available() else "cpu")
    assert args["workers"] == 0
    assert args["data"].endswith("dataset_final")


def test_metric_calculations() -> None:
    result = calculate_metrics([0, 1, 1, 2], [0, 1, 2, 2],
                               ("Blotch_Apple", "Normal_Apple", "Rot_Apple"))
    assert result["accuracy"] == 0.75
    assert len(result["confusion_matrix"]) == 3
    assert result["per_class"]["Normal_Apple"]["support"] == 2
    assert top_k_accuracy([0, 1], [[.6, .3, .1], [.2, .7, .1]], 1) == 1.0


def test_metrics_reject_misaligned_probabilities() -> None:
    with pytest.raises(ValueError):
        top_k_accuracy([0], [[.5, .5], [.5, .5]], 1)
