# Model Weights Documentation

This directory contains information and checkpoints for the trained YOLOv8 waste detection model.

---

## Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Model Filename** | `my_waste_model_6_class.pt` |
| **Architecture** | YOLOv8n (Nano) Detection Model |
| **Parameters** | 3,012,018 (~3.01 Million) |
| **Input Resolution** | $640 \times 640$ pixels |
| **File Size** | 5.96 MB (6,249,187 bytes) |
| **Number of Classes** | 6 (`cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`) |
| **Base Weights** | `yolov8n.pt` (MS COCO pretrained) |
| **Fine-tuning Epochs**| 40 epochs |

---

## Evaluation Benchmark Metrics (Validation Set)

- **mAP@50**: **76.46%** (`0.76463`)
- **mAP@50-95**: **56.18%** (`0.56178`)
- **Precision**: **73.74%** (`0.73736`)
- **Recall**: **69.71%** (`0.69708`)

---

## Obtaining the Trained Weights

Because GitHub recommends keeping repositories lightweight, model weights (`*.pt`) are hosted in GitHub Releases rather than committed directly to the git tree history.

### Option 1: Download from GitHub Releases
Download `my_waste_model_6_class.pt` from the latest repository release:
```bash
curl -L -o weights/my_waste_model_6_class.pt https://github.com/keshavbuilds23/waste-segregation-yolov8/releases/download/v1.0.0/my_waste_model_6_class.pt
```

### Option 2: Use Existing Local Checkpoint
Place `my_waste_model_6_class.pt` directly into this `weights/` directory.

---

## Quick Verification in Python

```python
from ultralytics import YOLO

model = YOLO("weights/my_waste_model_6_class.pt")
print("Classes:", model.names)
# Output: {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'organic', 4: 'paper', 5: 'plastic'}
```

