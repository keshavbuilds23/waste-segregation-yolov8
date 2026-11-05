# Dataset Documentation & Download Guide

This project uses the **Waste Classificator (v5)** dataset exported from Roboflow Universe.

- **Roboflow Universe URL**: [Waste Classificator Dataset on Roboflow](https://universe.roboflow.com/deep-learning-group-project/waste-classificator-mlabt/dataset/5)
- **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Total Images**: 4,525 images
- **Classes**: 6 categories (`cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`)

---

## Dataset Split Summary

| Split | Images | Annotations | Path |
| :--- | :---: | :---: | :--- |
| **Train** | 3,981 | 7,290 | `train/images/` & `train/labels/` |
| **Validation** | 328 | 524 | `valid/images/` & `valid/labels/` |
| **Test** | 216 | 396 | `test/images/` & `test/labels/` |
| **TOTAL** | **4,525** | **8,210** | |

---

## How to Download the Dataset

Because the full dataset comprises over 9,000 files and ~287 MB of disk space, it is excluded from git version control. You can download it directly using the Roboflow Python SDK or through the web interface.

### Option 1: Using the Roboflow Python SDK

1. Install the Roboflow library:
   ```bash
   pip install roboflow
   ```

2. Download and extract the dataset in YOLOv8 format:
   ```python
   from roboflow import Roboflow

   rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
   project = rf.workspace("deep-learning-group-project").project("waste-classificator-mlabt")
   version = project.version(5)
   dataset = version.download("yolov8")
   ```

### Option 2: Direct Web Download

1. Visit [Roboflow Universe - Waste Classificator v5](https://universe.roboflow.com/deep-learning-group-project/waste-classificator-mlabt/dataset/5).
2. Click **Download Dataset**.
3. Select format: **YOLOv8**.
4. Extract the ZIP archive into the project root directory so that `train/`, `valid/`, and `test/` folders reside alongside `data/`.

---

## Preprocessing & Data Augmentations Applied

- **Auto-Orientation**: EXIF-orientation stripped and standardized.
- **Resize**: Stretched to $640 \times 640$ pixels.
- **Rotations**: 90-degree rotations (none, clockwise, counter-clockwise with equal probability).
- **Brightness**: Random variation between -16% and +16%.
- **Noise**: Salt-and-pepper noise applied to 1.73% of pixels.

