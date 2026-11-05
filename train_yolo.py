"""
Waste Segregation YOLOv8 - Legacy Entry Point & Workflow Pipeline.
Provides backward-compatible train_model() and predict_and_get_percentage() functions
with robust path resolution, headless execution support, and clean English messaging.
"""

import os
import sys
from pathlib import Path

# Dependency check before importing ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("[INFO] 'ultralytics' library not found. Installing via pip...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        print("[SUCCESS] 'ultralytics' installed successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to install ultralytics automatically: {e}", file=sys.stderr)
        print("Please install requirements manually using: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    get_waste_category,
    compute_waste_statistics,
    format_statistics_report
)

DATA_CONFIG_PATH = str(PROJECT_ROOT / "data" / "data.yaml")
if not os.path.exists(DATA_CONFIG_PATH):
    DATA_CONFIG_PATH = str(PROJECT_ROOT / "data.yaml")

NUM_EPOCHS = 40

# Default sample test image
TEST_IMAGE_PATH = str(PROJECT_ROOT / "assets" / "sample_images" / "047.jpg")
if not os.path.exists(TEST_IMAGE_PATH):
    TEST_IMAGE_PATH = str(PROJECT_ROOT / "047.jpg")


def train_model(data_path: str = DATA_CONFIG_PATH, epochs: int = NUM_EPOCHS):
    """
    Trains YOLOv8 on the waste dataset using transfer learning from yolov8n.pt.
    """
    print("--- Starting YOLOv8 Training ---")

    if not os.path.exists(data_path):
        print(f"[ERROR] Data config file not found at: '{data_path}'", file=sys.stderr)
        print("Please verify that 'data.yaml' exists and image paths are correctly configured.", file=sys.stderr)
        return None

    model = YOLO("yolov8n.pt")

    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=640,
            batch=8
        )
    except Exception as e:
        print(f"[ERROR] An error occurred during training: {e}", file=sys.stderr)
        return None

    print("--- Training Complete ---")

    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(results.save_dir, "weights", "last.pt")

    final_model_path = str(PROJECT_ROOT / "weights" / "my_waste_model_6_class.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    import shutil
    try:
        shutil.copy2(best_model_path, final_model_path)
        # Also copy to root for legacy compatibility
        shutil.copy2(best_model_path, str(PROJECT_ROOT / "my_waste_model_6_class.pt"))
        print(f"[SUCCESS] Model saved as: {final_model_path}")
        return final_model_path
    except Exception as e:
        print(f"[WARNING] Error saving model to target path ({e}), using checkpoint path: {best_model_path}")
        return best_model_path


def predict_and_get_percentage(model_path: str, image_path: str):
    """
    Performs inference on an image and calculates Dry vs. Wet waste percentages.
    """
    print(f"\n--- Loading trained model from: {model_path} ---")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at '{model_path}'", file=sys.stderr)
        return None

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}", file=sys.stderr)
        return None

    print(f"--- Predicting on image: {image_path} ---")

    if not os.path.exists(image_path):
        print(f"[ERROR] Test image not found at '{image_path}'", file=sys.stderr)
        print("Please make sure the test image exists.", file=sys.stderr)
        return None

    try:
        results = model.predict(image_path)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        return None

    result = results[0]
    detected_items = []
    class_names = result.names

    print("\n--- Detected Items Report ---")
    if not result.boxes or len(result.boxes) == 0:
        print("No objects detected in the image.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = class_names[class_id]
            confidence = float(box.conf[0].item())
            category = get_waste_category(label)
            box_coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]

            detected_items.append({
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
                "category": category,
                "box": box_coords
            })

            print(f"  -> Found: {label:<10} (Classified as: {category}) [Confidence: {confidence:.2f}]")

    # Save output visualization
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_name = Path(image_path).name
    save_path = output_dir / f"annotated_{img_name}"
    result.save(filename=str(save_path))
    print(f"[INFO] Annotated output saved to: {save_path}")

    # Display image in GUI if possible (gracefully catches headless environments)
    try:
        result.show()
    except Exception as e:
        print(f"[INFO] GUI display unavailable ({e}). Annotated result was saved to disk.")

    stats = compute_waste_statistics(detected_items)
    print("\n--- Final Percentage Report ---")
    print(f"Total items detected: {stats['total_items']}")
    print(f"  Total Dry Waste items: {stats['dry_count']}")
    print(f"  Total Wet Waste items: {stats['wet_count']}")

    if stats["total_items"] > 0:
        print("\n===============================")
        print(f"  Dry Waste: {stats['dry_percent']:.2f}%")
        print(f"  Wet Waste: {stats['wet_percent']:.2f}%")
        print("===============================")
    else:
        print("No Dry or Wet waste items (as per our classes) were detected.")

    return stats


if __name__ == "__main__":
    trained_model_path = str(PROJECT_ROOT / "weights" / "my_waste_model_6_class.pt")
    if not os.path.exists(trained_model_path):
        trained_model_path = str(PROJECT_ROOT / "my_waste_model_6_class.pt")

    if not os.path.exists(trained_model_path):
        print(f"No trained model found at '{trained_model_path}'.")
        print("Starting training first...")

        if not os.path.exists(DATA_CONFIG_PATH):
            print(f"[ERROR] Data config file not found at '{DATA_CONFIG_PATH}'", file=sys.stderr)
        else:
            model_path = train_model(DATA_CONFIG_PATH)
            if model_path:
                predict_and_get_percentage(model_path, TEST_IMAGE_PATH)
    else:
        print(f"Found trained model: {trained_model_path}")
        predict_and_get_percentage(trained_model_path, TEST_IMAGE_PATH)
