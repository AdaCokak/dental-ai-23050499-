from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionSettings:
    base_conf: float = 0.30
    low_conf: float = 0.15
    min_area: float = 5000.0
    iou_threshold: float = 0.10
    imgsz: int = 640


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / (box_area(box_a) + box_area(box_b) - inter)


def extract_recall_biased_boxes(model, image_path: str, settings: ExtractionSettings) -> list[list[float]]:
    return [item["box"] for item in extract_recall_biased_candidates(model, image_path, settings)]


def extract_recall_biased_candidates(model, image_path: str, settings: ExtractionSettings) -> list[dict[str, object]]:
    high = model.predict(source=image_path, conf=settings.base_conf, imgsz=settings.imgsz, verbose=False)[0]
    low = model.predict(source=image_path, conf=settings.low_conf, imgsz=settings.imgsz, verbose=False)[0]

    selected = []
    if high.boxes is not None:
        for box, conf in zip(high.boxes.xyxy.tolist(), high.boxes.conf.tolist()):
            selected.append({"box": box, "source": "high", "confidence": conf, "area": box_area(box)})
    low_boxes = []
    if low.boxes is not None:
        for box, conf in zip(low.boxes.xyxy.tolist(), low.boxes.conf.tolist()):
            low_boxes.append({"box": box, "confidence": conf, "area": box_area(box)})

    for candidate in low_boxes:
        if candidate["area"] < settings.min_area:
            continue
        if any(box_iou(candidate["box"], existing["box"]) > settings.iou_threshold for existing in selected):
            continue
        selected.append(
            {
                "box": candidate["box"],
                "source": "low",
                "confidence": candidate["confidence"],
                "area": candidate["area"],
            }
        )

    return selected
