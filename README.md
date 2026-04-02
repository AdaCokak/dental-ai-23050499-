# Dental AI — Panoramic X-Ray Disease Detection

A Streamlit application that detects dental pathologies (caries, deep caries, impacted teeth) in panoramic X-ray images using YOLOv11 instance segmentation.

## Models

| Model | Training data | mAP50 |
|---|---|---|
| **Fine-tuned (primary)** | DENTEX (577 train) + 49 clinical films | 60.2% on DENTEX val |
| YOLO V2 baseline | DENTEX 577 train images | 66.8% on DENTEX val |
| Tooth segmenter | 598 panoramic polygons (archive dataset) | — |

The **fine-tuned model is the default** in the app. It adapts the V2 baseline to real clinical scanner images.

## Test images

The `TEST/` folder contains 6 real clinical panoramic X-rays for testing the app. These are the images the tutor should use to evaluate the cavity detection.

To test:
1. Run the app (see below)
2. Enable **"Use my own image"** toggle
3. Upload any image from the `TEST/` folder
4. Select **"Detect Cavities — Fine-tuned on Clinical Films"** and press **Run**

## Setup

Python 3.11 required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app

```bash
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## App features

- **Detect Cavities — Fine-tuned on Clinical Films** (default): runs the fine-tuned YOLOv11s-seg model
- **Detect Cavities — YOLO V2 (Panoramic)**: runs the V2 baseline trained on DENTEX only
- **Show labeled teeth**: overlays ground-truth polygon annotations (requires dataset — see below)
- **Find teeth only**: runs the tooth segmentation model to count and locate teeth (requires dataset)

Hover over any detection result image to hide the overlay and see the original X-ray.

> **Note on the built-in dataset:** The "Show labeled teeth" and "Find teeth only" options use panoramic images from `archive/Teeth Segmentation JSON/d2/` (598 images, ~465 MB). This folder is not included in the repository due to size. To use these options, download the dataset separately and place it at that path. **The two cavity detection options work with any uploaded image — enable "Use my own image" to upload your own panoramic X-ray without needing the dataset.**

## Key files

```
app.py                                           — Streamlit UI
annotate.py                                      — annotation tool (used to label clinical films)
src/dental_ai/
  dentex_infer.py                                — YOLOv11 inference + colored mask rendering
  extraction.py                                  — tooth candidate extraction (tooth-only mode)
  train.py                                       — YOLO training script
runs/segment/
  dentex_v2_finetuned/weights/best.pt            — fine-tuned weights (primary)
  dentex_disease_seg_v2/weights/best.pt          — V2 baseline weights
  archive_tooth_seg/weights/best.pt              — tooth segmentation weights
TEST/                                            — 6 clinical X-rays for testing
```

## Disease classes

| ID | Class | Color |
|---|---|---|
| 0 | Impacted | Blue |
| 1 | Caries | Orange |
| 2 | Caries (periapical) | Yellow |
| 3 | Deep Caries | Red |

## Hardware

Tested on Apple Silicon (MPS). Works on any machine with PyTorch support (CPU, CUDA, MPS).
