from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageColor, ImageDraw

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "archive" / "Teeth Segmentation JSON" / "d2"
META_PATH = PROJECT_ROOT / "archive" / "Teeth Segmentation JSON" / "meta.json"
DEFAULT_TOOTH_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "archive_tooth_seg" / "weights" / "best.pt"
DEFAULT_DENTEX_V2_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "dentex_disease_seg_v2" / "weights" / "best.pt"
DEFAULT_FINETUNED_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "dentex_v2_finetuned" / "weights" / "best.pt"


def clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def show_hover_image(overlay: Image.Image, original: Image.Image) -> None:
    """Show overlay image; hovering reveals the clean original underneath."""
    ov_b64 = _img_to_b64(overlay)
    or_b64 = _img_to_b64(original)
    st.components.v1.html(f"""
    <style>
      .hov-wrap {{
        position: relative;
        width: 100%;
        cursor: crosshair;
        user-select: none;
      }}
      .hov-wrap img {{
        width: 100%;
        display: block;
        border-radius: 6px;
        transition: opacity 0.15s ease;
      }}
      .hov-wrap .hov-orig {{
        position: absolute;
        top: 0; left: 0;
        opacity: 0;
      }}
      .hov-wrap:hover .hov-orig {{
        opacity: 1;
      }}
      .hov-wrap:hover .hov-overlay {{
        opacity: 0;
      }}
      .hov-label {{
        font-size: 12px;
        color: #888;
        margin-top: 4px;
        text-align: center;
      }}
    </style>
    <div class="hov-wrap">
      <img class="hov-overlay" src="data:image/jpeg;base64,{ov_b64}" />
      <img class="hov-orig"    src="data:image/jpeg;base64,{or_b64}" />
    </div>
    <div class="hov-label">Hover to hide detections</div>
    """, height=int(overlay.height * 900 / overlay.width) + 30)


@st.cache_data
def load_meta() -> tuple[dict[int, str], dict[str, str]]:
    with META_PATH.open() as handle:
        meta = json.load(handle)
    class_map = {entry["id"]: entry["title"] for entry in meta["classes"]}
    color_map = {entry["title"]: entry["color"] for entry in meta["classes"]}
    return class_map, color_map


@st.cache_data
def dataset_images() -> list[Path]:
    return sorted((DATASET_ROOT / "img").glob("*.jpg"), key=lambda path: int(path.stem))


@st.cache_resource
def load_tooth_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed in this environment.")
    return YOLO(weights_path)


@st.cache_resource
def load_dentex_model(weights_path: str):
    from src.dental_ai.dentex_infer import load_dentex_model as _load

    return _load(weights_path)


@st.cache_resource
def load_finetuned_model(weights_path: str):
    from src.dental_ai.dentex_infer import load_dentex_model as _load
    return _load(weights_path)


def load_image(image_path: Path | None, upload) -> Image.Image | None:
    if upload is not None:
        return Image.open(upload).convert("RGB")
    if image_path is not None:
        return Image.open(image_path).convert("RGB")
    return None


def draw_ground_truth(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    annotation_path = DATASET_ROOT / "ann" / f"{image_path.name}.json"
    class_map, color_map = load_meta()
    with annotation_path.open() as handle:
        annotation = json.load(handle)

    for obj in annotation.get("objects", []):
        if obj.get("geometryType") != "polygon":
            continue
        label = class_map.get(obj["classId"], "tooth")
        color = color_map.get(label, "#00FF00")
        rgb = ImageColor.getrgb(color)
        points = []
        for point in obj["points"]["exterior"]:
            if isinstance(point, dict):
                points.append((point["x"], point["y"]))
            else:
                points.append((point[0], point[1]))
        if len(points) >= 3:
            draw.polygon(points, outline=rgb + (255,), fill=rgb + (40,))
    return image


def run_tooth_only(image: Image.Image, confidence: float) -> tuple[Image.Image, dict[str, int]]:
    from src.dental_ai.extraction import ExtractionSettings, extract_recall_biased_candidates

    temp_path = PROJECT_ROOT / "tmp_streamlit_input.png"
    image.save(temp_path)

    model = load_tooth_model(str(DEFAULT_TOOTH_WEIGHTS))
    candidates = extract_recall_biased_candidates(
        model,
        str(temp_path),
        ExtractionSettings(imgsz=1024, base_conf=max(confidence, 0.30), low_conf=min(confidence, 0.15)),
    )

    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for index, candidate in enumerate(candidates):
        box = candidate["box"]
        draw.rectangle(box, outline=(55, 200, 90), width=3)
        draw.text((box[0], box[1]), str(index + 1), fill=(55, 200, 90))

    summary = {
        "teeth_found": len(candidates),
        "high_conf_teeth": sum(1 for item in candidates if item["source"] == "high"),
        "low_conf_extra_teeth": sum(1 for item in candidates if item["source"] == "low"),
    }
    return rendered, summary


def show_summary_box(title: str, items: dict[str, int]) -> None:
    st.subheader(title)
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items.items()):
        col.metric(label.replace("_", " ").title(), value)


def main() -> None:
    st.set_page_config(page_title="Dental AI", layout="wide")
    st.title("Dental X-Ray AI")
    st.write("Use the controls below, then press one button. The app will either show teeth or run the lesion analysis.")

    available_images = dataset_images()
    left, right = st.columns([1, 2])

    with left:
        st.subheader("1. Choose Image")
        use_upload = st.toggle("Use my own image", value=False)
        uploaded_file = st.file_uploader("Upload panoramic image", type=["jpg", "jpeg", "png"]) if use_upload else None

        selected_path = None
        if not use_upload:
            image_names = [path.name for path in available_images]
            selected_name = st.selectbox("Dataset image", image_names, index=0)
            selected_path = DATASET_ROOT / "img" / selected_name

        st.subheader("2. Pick Action")
        action = st.radio(
            "What do you want to see?",
            [
                "Detect Cavities — Fine-tuned on Clinical Films",
                "Detect Cavities — YOLO V2 (Panoramic)",
                "Show labeled teeth",
                "Find teeth only",
            ],
        )

        confidence = st.slider("Detection confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        run_clicked = st.button("Run", use_container_width=True, type="primary")

        if action == "Detect Cavities — Fine-tuned on Clinical Films":
            st.info(
                "**Final model — V2 fine-tuned on 49 annotated clinical panoramics.** "
                "Adapted to real clinical scanner data.\n\n"
                "**Color key:** 🟠 Caries  🔴 Deep Caries  🔵 Impacted"
            )
            if not DEFAULT_FINETUNED_WEIGHTS.exists():
                st.warning("Fine-tuned weights not found.")
        if action == "Detect Cavities — YOLO V2 (Panoramic)":
            st.info(
                "**YOLOv11s trained on DENTEX MICCAI 2023.** Baseline model. "
                "mAP50: 66.8%\n\n"
                "**Color key:** 🟠 Caries  🔴 Deep Caries  🟡 Caries (periapical)  🔵 Impacted"
            )
            if not DEFAULT_DENTEX_V2_WEIGHTS.exists():
                st.warning("DENTEX V2 model weights not found.")

    image = load_image(selected_path, uploaded_file)

    with right:
        if image is None:
            st.info("Choose a dataset image or upload your own panoramic image.")
            return

        if not run_clicked:
            st.subheader("Preview")
            st.image(image, use_container_width=True)
            return

        if action == "Detect Cavities — YOLO V2 (Panoramic)":
            if not DEFAULT_DENTEX_V2_WEIGHTS.exists():
                st.error("DENTEX V2 model weights not found.")
                return
            with st.spinner("Running YOLO V2 cavity detection…"):
                from src.dental_ai.dentex_infer import run_dentex_inference
                model = load_dentex_model(str(DEFAULT_DENTEX_V2_WEIGHTS))
                result = run_dentex_inference(image, model, confidence=confidence)
                overlay = result.overlay
                summary = {
                    "caries": result.caries_count,
                    "deep_caries": result.deep_caries_count,
                    "impacted": result.impacted_count,
                }
                rows = [
                    {"#": i+1, "Condition": d.class_name, "Confidence": f"{d.confidence:.1%}"}
                    for i, d in enumerate(result.detections)
                ]
            show_summary_box("Detected Conditions (V2 Baseline)", summary)
            show_hover_image(overlay, image)
            if rows:
                st.subheader("Detection Details")
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.success("No diseased teeth detected at this confidence level. Try lowering the detection confidence slider.")
            return

        if action == "Detect Cavities — Fine-tuned on Clinical Films":
            if not DEFAULT_FINETUNED_WEIGHTS.exists():
                st.error("Fine-tuned weights not found.")
                return
            with st.spinner("Running fine-tuned model…"):
                from src.dental_ai.dentex_infer import run_dentex_inference
                model = load_finetuned_model(str(DEFAULT_FINETUNED_WEIGHTS))
                result = run_dentex_inference(image, model, confidence=confidence)
            summary = {
                "caries": result.caries_count,
                "deep_caries": result.deep_caries_count,
                "impacted": result.impacted_count,
            }
            show_summary_box("Detected Conditions (Fine-tuned)", summary)
            show_hover_image(result.overlay, image)
            if result.detections:
                rows = [
                    {"#": i+1, "Condition": d.class_name, "Confidence": f"{d.confidence:.1%}"}
                    for i, d in enumerate(result.detections)
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.success("No findings at this confidence level. Try lowering the slider.")
            return

        if action == "Show labeled teeth":
            if selected_path is None:
                st.info("Labeled teeth view only works with built-in dataset images.")
                st.image(image, use_container_width=True)
                return
            labeled = draw_ground_truth(selected_path)
            st.subheader("Labeled Teeth")
            st.image(labeled, use_container_width=True)
            return

        if action == "Find teeth only":
            if not DEFAULT_TOOTH_WEIGHTS.exists():
                st.error("Tooth model weights are missing.")
                return
            result, summary = run_tooth_only(image, confidence)
            show_summary_box("Result", summary)
            st.image(result, use_container_width=True)
            return



if __name__ == "__main__":
    main()
