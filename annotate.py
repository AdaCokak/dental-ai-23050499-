"""Dental X-Ray Annotation Tool

Lets you upload panoramic X-ray images, draw polygon labels for dental diseases,
and save them directly in YOLO segmentation format ready for fine-tuning.

Run on port 8502 alongside the main app:
    streamlit run annotate.py --server.port 8502

Output:
    data/annotations_finetune/
        images/   ← copies of annotated images
        labels/   ← YOLO segmentation .txt files
        dataset.yaml
"""

from __future__ import annotations

import base64
import io
import json
import shutil
from hashlib import md5
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.elements.image as _st_image

# ── Compatibility patch ───────────────────────────────────────────────────────
# streamlit-drawable-canvas calls st.elements.image.image_to_url which was
# removed in Streamlit >=1.45. Patch it back by registering the image with
# Streamlit's MediaFileManager so the React canvas receives a real server URL.
if not hasattr(_st_image, "image_to_url"):
    def _image_to_url(image, width, clamp, channels, output_format, image_id):
        from PIL import Image as _PILImage
        buf = io.BytesIO()
        if not isinstance(image, _PILImage.Image):
            image = _PILImage.fromarray(image)
        image.save(buf, format=output_format or "PNG")
        data = buf.getvalue()
        try:
            from streamlit.runtime import get_instance as _get_rt
            mgr = _get_rt().media_file_mgr
            url = mgr.add(data, "image/png", image_id)
            return url  # already starts with /
        except Exception:
            # Absolute fallback: base64 (canvas won't render but avoids crash)
            b64 = base64.b64encode(data).decode()
            return f"data:image/png;base64,{b64}"
    _st_image.image_to_url = _image_to_url
# ─────────────────────────────────────────────────────────────────────────────

from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

# ── Classes ──────────────────────────────────────────────────────────────────
# 0-2 match V2 training exactly. Pulp (3) is an anatomy class — teaches the
# model "this dark region = pulp, NOT caries". Filtered from disease output
# at inference time. Can be excluded from fine-tuning config if not needed.
CLASSES   = {0: "Impacted", 1: "Caries", 2: "Deep Caries", 3: "Pulp"}
COLORS    = {0: "#3B82F6", 1: "#F97316", 2: "#EF4444", 3: "#14B8A6"}
#            blue            orange         red             teal
BG_COLOR  = "#000000"

OUTPUT_DIR   = Path("data/annotations_finetune")
IMAGES_DIR   = OUTPUT_DIR / "images"
LABELS_DIR   = OUTPUT_DIR / "labels"
CANVAS_W     = 900    # display width of canvas (fits Streamlit wide layout with sidebar)


# ── Helpers ───────────────────────────────────────────────────────────────────

def yolo_line_to_canvas(line: str, img_w: int, img_h: int,
                         scale_x: float, scale_y: float) -> dict | None:
    """Convert a YOLO seg line back to a canvas-compatible rect/path object."""
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    class_id = int(parts[0])
    coords   = list(map(float, parts[1:]))
    path     = []
    for i in range(0, len(coords), 2):
        px = coords[i]   * img_w * scale_x
        py = coords[i+1] * img_h * scale_y
        path.append(["L" if i > 0 else "M", px, py])
    path.append(["z"])
    color = COLORS.get(class_id, "#FFFFFF")
    return {
        "type": "path",
        "version": "5.3.0",
        "originX": "left",
        "originY": "top",
        "left": 0,
        "top": 0,
        "width": CANVAS_W,
        "height": 9999,
        "fill": color + "33",
        "stroke": color,
        "strokeWidth": 2,
        "path": path,
    }


def canvas_path_to_yolo(path: list, img_w: int, img_h: int,
                          scale_x: float, scale_y: float) -> list[float] | None:
    """Extract normalized polygon coords from a canvas path object.

    Handles both polygon paths (M/L) and freedraw paths (M/Q/C).
    For curves, only the endpoint of each segment is used.
    Downsamples to at most 64 points to keep YOLO files compact.
    """
    raw = []
    for cmd in path:
        if not cmd:
            continue
        t = cmd[0]
        if t == "M" and len(cmd) == 3:
            raw.append((cmd[1], cmd[2]))
        elif t == "L" and len(cmd) == 3:
            raw.append((cmd[1], cmd[2]))
        elif t == "Q" and len(cmd) == 5:          # quadratic bezier → endpoint
            raw.append((cmd[3], cmd[4]))
        elif t == "C" and len(cmd) == 7:          # cubic bezier → endpoint
            raw.append((cmd[5], cmd[6]))

    if len(raw) < 3:
        return None

    # Downsample to ≤64 points (YOLO doesn't need high-res polygons)
    if len(raw) > 64:
        step = len(raw) / 64
        raw = [raw[int(i * step)] for i in range(64)]

    cw = img_w * scale_x
    ch = img_h * scale_y
    coords = []
    for x, y in raw:
        coords.extend([max(0.0, min(1.0, x / cw)),
                       max(0.0, min(1.0, y / ch))])
    return coords


def load_existing_annotations(label_path: Path, img_w: int, img_h: int,
                                scale_x: float, scale_y: float) -> list[dict]:
    objects = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            obj = yolo_line_to_canvas(line, img_w, img_h, scale_x, scale_y)
            if obj:
                objects.append(obj)
    return objects


def save_annotations(label_path: Path, annotations: list[dict],
                      class_id: int, img_w: int, img_h: int,
                      scale_x: float, scale_y: float) -> int:
    """Append new canvas shapes to the label file. Returns count saved."""
    saved = 0
    with label_path.open("a") as f:
        for obj in annotations:
            if obj.get("type") != "path":
                continue
            coords = canvas_path_to_yolo(
                obj["path"], img_w, img_h, scale_x, scale_y)
            if coords is None:
                continue
            coords_str = " ".join(f"{v:.6f}" for v in coords)
            f.write(f"{class_id} {coords_str}\n")
            saved += 1
    return saved


def count_class_labels(label_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in CLASSES.values()}
    if not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            name = CLASSES.get(int(parts[0]))
            if name:
                counts[name] += 1
    return counts


def write_dataset_yaml() -> None:
    content = f"""path: {OUTPUT_DIR.resolve()}
train: images
val: images

nc: 4
names: ['Impacted', 'Caries', 'Deep Caries', 'Pulp']
# Pulp (class 3) is an anatomy label — filter from disease output at inference.
# To fine-tune without Pulp: use dentex_v2_finetune.yaml (nc=3, ignores class 3).
# To fine-tune with Pulp:    use dentex_v2_finetune_pulp.yaml (nc=4).
"""
    (OUTPUT_DIR / "dataset.yaml").write_text(content)


def render_review(image: Image.Image, label_path: Path) -> Image.Image:
    """Draw all saved annotations on the image for review."""
    from PIL import ImageFont
    overlay = image.convert("RGBA")
    draw_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    img_w, img_h = image.size

    if not label_path.exists():
        return image

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        # Convert normalized coords to pixel coords
        pts = []
        for i in range(0, len(coords), 2):
            pts.append((int(coords[i] * img_w), int(coords[i+1] * img_h)))

        hex_color = COLORS.get(class_id, "#FFFFFF").lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

        draw.polygon(pts, fill=(r, g, b, 60), outline=(r, g, b, 220))
        draw.line(pts + [pts[0]], fill=(r, g, b, 220), width=3)

        # Label at centroid
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        label = CLASSES.get(class_id, str(class_id))
        draw.text((cx, cy), label, fill=(r, g, b, 255))

    result = Image.alpha_composite(overlay, draw_layer)
    return result.convert("RGB")


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Dental Annotator", layout="wide")
    st.title("Dental X-Ray Annotation Tool")
    st.caption("Draw polygon labels → save as YOLO segmentation format for fine-tuning")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("1. Upload Image")
        uploaded = st.file_uploader(
            "Upload panoramic X-ray",
            type=["jpg", "jpeg", "png"],
            key="uploader",
        )

        st.header("2. Select Class")
        class_id = st.radio(
            "Disease class to draw",
            options=list(CLASSES.keys()),
            format_func=lambda x: f"{CLASSES[x]}  (class {x})",
            index=1,    # default: Caries
        )
        st.markdown(
            f"<div style='width:100%;height:8px;background:{COLORS[class_id]};"
            f"border-radius:4px;margin-bottom:8px'></div>",
            unsafe_allow_html=True,
        )

        st.header("3. Draw & Save")
        st.info(
            "**Draw freely** around the lesion — just circle it like a pen.\n\n"
            "Each closed loop = one annotation.\n\n"
            "Press **Save polygons** after drawing each batch."
        )
        save_btn      = st.button("💾  Save polygons", use_container_width=True,
                                   type="primary")
        clear_btn     = st.button("🗑  Clear label file (start over)",
                                   use_container_width=True)

        st.header("4. Progress")
        annotated = sorted(LABELS_DIR.glob("*.txt"))
        st.metric("Images annotated", len(annotated))
        if annotated:
            st.write("**Annotated files:**")
            for p in annotated[-10:]:
                counts = count_class_labels(p)
                parts  = [f"{v}× {k}" for k, v in counts.items() if v > 0]
                st.write(f"- `{p.stem}`: {', '.join(parts) if parts else 'empty'}")

        st.markdown("---")
        if st.button("📄  Regenerate dataset.yaml"):
            write_dataset_yaml()
            st.success(f"Saved to {OUTPUT_DIR}/dataset.yaml")

    # ── Main panel ────────────────────────────────────────────────────────────
    if uploaded is None:
        st.info("Upload a panoramic X-ray in the sidebar to start annotating.")
        return

    image = Image.open(uploaded).convert("RGB")
    img_w, img_h = image.size

    # Scale image to canvas width
    scale_x = CANVAS_W / img_w
    scale_y = scale_x                        # keep aspect ratio
    canvas_h = int(img_h * scale_y)
    display_img = image.resize((CANVAS_W, canvas_h), Image.LANCZOS)

    stem       = Path(uploaded.name).stem
    label_path = LABELS_DIR / f"{stem}.txt"
    image_path = IMAGES_DIR / uploaded.name

    # Load existing annotations to show on canvas
    existing = load_existing_annotations(
        label_path, img_w, img_h, scale_x, scale_y)
    initial_drawing = {"version": "5.3.0", "objects": existing} if existing else None

    # ── Current annotation summary ────────────────────────────────────────────
    counts = count_class_labels(label_path)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Impacted",    counts["Impacted"],    delta=None)
    col2.metric("Caries",      counts["Caries"],      delta=None)
    col3.metric("Deep Caries", counts["Deep Caries"], delta=None)
    col4.metric("Pulp",        counts["Pulp"],        delta=None)

    st.markdown(
        f"**File:** `{uploaded.name}` &nbsp;|&nbsp; "
        f"**Size:** {img_w}×{img_h} px &nbsp;|&nbsp; "
        f"**Drawing:** <span style='color:{COLORS[class_id]}'>"
        f"**{CLASSES[class_id]}**</span>",
        unsafe_allow_html=True,
    )

    # ── Tabs: Draw vs Review ──────────────────────────────────────────────────
    tab_draw, tab_review = st.tabs(["✏️  Draw", "🔍  Review Annotations"])

    with tab_review:
        if not label_path.exists() or label_path.stat().st_size == 0:
            st.info("No annotations saved yet for this image.")
        else:
            review_img = render_review(image, label_path)
            st.image(review_img, use_container_width=True)
            legend_cols = st.columns(len(CLASSES))
            for col, (cid, name) in zip(legend_cols, CLASSES.items()):
                col.markdown(
                    f"<span style='color:{COLORS[cid]};font-weight:bold'>■ {name}</span>",
                    unsafe_allow_html=True,
                )

    with tab_draw:
        # ── Canvas ───────────────────────────────────────────────────────────
        canvas_result = st_canvas(
            fill_color=COLORS[class_id] + "44",   # semi-transparent fill
            stroke_width=3,
            stroke_color=COLORS[class_id],
            background_image=display_img,
            background_color=BG_COLOR,
            update_streamlit=True,
            height=canvas_h,
            width=CANVAS_W,
            drawing_mode="freedraw",
            initial_drawing=initial_drawing,
            display_toolbar=True,
            key=f"canvas_{stem}_{class_id}",
        )

        # ── Save ─────────────────────────────────────────────────────────────
        if save_btn:
            if canvas_result.json_data is None:
                st.warning("Nothing to save — draw some polygons first.")
            else:
                objects = canvas_result.json_data.get("objects", [])
                # Only save NEW polygons (not already-loaded ones)
                new_objects = objects[len(existing):]

                if not new_objects:
                    st.warning("No new polygons to save. Draw something first.")
                else:
                    # Copy image if not already saved
                    if not image_path.exists():
                        image.save(image_path)

                    n = save_annotations(
                        label_path, new_objects, class_id,
                        img_w, img_h, scale_x, scale_y,
                    )
                    write_dataset_yaml()
                    if n > 0:
                        st.success(
                            f"Saved {n} **{CLASSES[class_id]}** polygon(s) "
                            f"to `{label_path.name}` ✓"
                        )
                        st.rerun()
                    else:
                        st.warning(
                            "Polygons drawn but couldn't extract coordinates. "
                            "Try drawing a larger shape."
                        )

        if clear_btn:
            if label_path.exists():
                label_path.unlink()
                st.success(f"Cleared `{label_path.name}`")
                st.rerun()

    # ── Instructions ─────────────────────────────────────────────────────────
    with st.expander("How to use"):
        st.markdown("""
**Drawing:**
1. Select the disease class in the sidebar
2. Click and drag to draw freely around the lesion
3. Release — each stroke becomes one annotation shape
4. Draw as many shapes as needed for the current class
5. Click **Save polygons** — then switch class and repeat

**Per-tooth annotation order (recommended):**
For each tooth: draw **Pulp** first (teal) → then **Caries/Deep Caries** if present

**Class tips:**
- 🔵 **Impacted** — outline the full buried/angled tooth
- 🟠 **Caries** — trace the dark demineralized area at the tooth surface
- 🔴 **Deep Caries** — trace the deeper dark zone nearing the pulp chamber
- 🩵 **Pulp** — rough oval outline of the dark pulp chamber at tooth center.
  Does NOT need to be precise — just enough for the model to learn
  "dark center region = pulp, not caries". Quick rough outline is fine.

**Pulp annotation strategy:**
- Annotate pulp on EVERY tooth in the image (healthy or not)
- This teaches the AI: dark center oval = pulp → ignore for caries detection
- One polygon per tooth pulp chamber (not root canals)

**Fine-tuning options:**
- **With pulp class** → `configs/dentex_v2_finetune_pulp.yaml` (recommended if you annotated pulp)
- **Without pulp class** → `configs/dentex_v2_finetune.yaml` (3 classes only)

**Output format:**
YOLO segmentation `.txt` in `data/annotations_finetune/labels/`
Each line: `class_id x1 y1 x2 y2 ... xn yn` (normalized 0–1)
        """)


if __name__ == "__main__":
    main()
