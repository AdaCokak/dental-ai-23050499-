"""Inference module for the DENTEX disease segmentation model.

Runs YOLOv11 instance segmentation on a panoramic X-ray and draws
color-coded masks over detected diseased teeth.

Disease classes (matching prepare_dentex_yolo.py):
    0: Impacted          — blue
    1: Caries            — red-orange
    2: Periapical Lesion — yellow
    3: Deep Caries       — bright red
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENTEX_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "dentex_disease_seg" / "weights" / "best.pt"

CLASS_NAMES = ["Impacted", "Caries", "Caries", "Deep Caries"]

# RGBA fill and outline colors per class
CLASS_COLORS: dict[int, tuple[int, int, int, int]] = {
    0: (80,  130, 220, 160),   # Impacted — blue
    1: (240, 100,  40, 160),   # Caries — orange-red
    2: (230, 210,  30, 160),   # Periapical Lesion — yellow
    3: (220,  30,  30, 180),   # Deep Caries — bright red
}
CLASS_OUTLINE: dict[int, tuple[int, int, int, int]] = {
    0: ( 60, 100, 200, 230),
    1: (220,  60,  10, 230),
    2: (200, 180,   0, 230),
    3: (200,   0,   0, 230),
}


@dataclass
class DetectedTooth:
    class_id: int
    class_name: str
    confidence: float
    box: list[float]      # [x1, y1, x2, y2] in pixels
    mask_polygon: list[tuple[int, int]] | None = None


@dataclass
class DentexResult:
    overlay: Image.Image
    detections: list[DetectedTooth] = field(default_factory=list)
    caries_count: int = 0
    deep_caries_count: int = 0
    periapical_count: int = 0
    impacted_count: int = 0


def apply_clahe(image: Image.Image) -> Image.Image:
    """Apply CLAHE to enhance contrast in a dental X-ray before inference.

    Converts to LAB color space, applies CLAHE only to the L (luminance)
    channel (tile 16×16, clipLimit 2.0), then converts back to RGB.
    This preserves color balance while enhancing local contrast in cavity
    borders, periapical shadows, and bone texture.
    """
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_dentex_model(weights_path: str | Path):
    """Load a trained DENTEX YOLO model. Returns the YOLO model object."""
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed.")
    return YOLO(str(weights_path))


def run_dentex_inference(
    image: Image.Image,
    model,
    confidence: float = 0.25,
    iou: float = 0.40,
    imgsz: int = 1280,
) -> DentexResult:
    """Run DENTEX disease detection on a panoramic X-ray.

    Returns a DentexResult with:
    - overlay: the original image with color-coded disease masks drawn on top
    - detections: list of DetectedTooth objects
    - per-class counts
    """
    import tempfile, os

    # Save to temp file for YOLO (handles large PIL images reliably)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        results = model.predict(
            source=tmp_path,
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
    finally:
        os.unlink(tmp_path)

    result = results[0]
    detections: list[DetectedTooth] = []

    # --- Draw overlay ---
    overlay = image.convert("RGBA")
    mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    box_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)
    box_draw = ImageDraw.Draw(box_layer)

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().int().numpy()

        masks_xy = None
        if result.masks is not None:
            masks_xy = result.masks.xy  # list of numpy arrays of shape (N,2)

        for i, (box, conf, cid) in enumerate(zip(boxes_xyxy, confs, class_ids)):
            cid = int(cid)
            x1, y1, x2, y2 = box.tolist()
            fill = CLASS_COLORS.get(cid, (200, 200, 200, 140))
            outline = CLASS_OUTLINE.get(cid, (150, 150, 150, 230))
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)

            poly_points = None
            if masks_xy is not None and i < len(masks_xy):
                pts = masks_xy[i]
                if pts is not None and len(pts) >= 3:
                    poly_points = [(int(p[0]), int(p[1])) for p in pts]
                    mask_draw.polygon(poly_points, fill=fill)

            # Bounding box outline
            box_draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)

            # Label tag (clamped so it never renders above y=0)
            label_text = f"{name} {conf:.0%}"
            tag_top = max(0, y1 - 18)
            tag_bottom = tag_top + 18
            box_draw.rectangle([x1, tag_top, x1 + len(label_text) * 7 + 4, tag_bottom], fill=outline)
            box_draw.text((x1 + 2, tag_top + 2), label_text, fill=(255, 255, 255, 255))

            detections.append(
                DetectedTooth(
                    class_id=cid,
                    class_name=name,
                    confidence=float(conf),
                    box=[x1, y1, x2, y2],
                    mask_polygon=poly_points,
                )
            )

    overlay = Image.alpha_composite(overlay, mask_layer)
    overlay = Image.alpha_composite(overlay, box_layer).convert("RGB")

    counts = {cid: sum(1 for d in detections if d.class_id == cid) for cid in range(4)}
    return DentexResult(
        overlay=overlay,
        detections=detections,
        caries_count=counts.get(1, 0),
        deep_caries_count=counts.get(3, 0),
        periapical_count=counts.get(2, 0),
        impacted_count=counts.get(0, 0),
    )
