from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "archive_seg.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tooth segmentation model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config file.")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    """Return the requested device if available, otherwise fall back to cpu."""
    import torch

    if requested == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main() -> None:
    args = parse_args()
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)

    dataset_yaml = PROJECT_ROOT / config["dataset_root"] / "dataset.yaml"
    device = _resolve_device(config.get("device", "cpu"))
    model = YOLO(config["model"])
    model.train(
        data=str(dataset_yaml),
        imgsz=config["image_size"],
        epochs=config["epochs"],
        batch=config["batch"],
        device=device,
        workers=config.get("workers", 4),
        copy_paste=config.get("copy_paste", 0.0),
        degrees=config.get("degrees", 0.0),
        mixup=config.get("mixup", 0.0),
        close_mosaic=config.get("close_mosaic", 10),
        lr0=config.get("lr0", 0.01),
        lrf=config.get("lrf", 0.01),
        freeze=config.get("freeze", None),
        patience=config.get("patience", 50),
        amp=config.get("amp", True),
        project=str(PROJECT_ROOT / "runs" / "segment"),
        name=config.get("project_name", "archive_tooth_seg"),
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
