"""
Inference and Waste Segregation Analysis Module.
Performs object detection using fine-tuned YOLOv8, classifies items into 6 classes,
and generates Dry vs. Wet waste composition percentages.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: 'ultralytics' is not installed in the active environment.\n"
        "Please install required dependencies via:\n"
        "    pip install -r requirements.txt\n"
        "or: pip install ultralytics",
        file=sys.stderr
    )
    sys.exit(1)

# Ensure local imports work regardless of execution location
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    PROJECT_ROOT,
    get_waste_category,
    compute_waste_statistics,
    format_statistics_report,
    resolve_path
)


def predict_waste(
    model_path: str,
    source: str,
    conf: float = 0.25,
    save_dir: Optional[str] = None,
    show: bool = False,
    device: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Runs YOLOv8 object detection on a source image/video, computes segregation stats,
    and returns detected items.

    Args:
        model_path: Path to .pt model checkpoint.
        source: Image path, directory, or video file.
        conf: Minimum confidence threshold.
        save_dir: Directory where annotated output images should be saved.
        show: Whether to display interactive desktop window.
        device: Hardware device ('cpu', 'cuda', '0', etc.)

    Returns:
        List of detection summary dictionaries.
    """
    model_file = resolve_path(model_path, "weights/my_waste_model_6_class.pt")
    if not model_file.exists():
        # Fallback to root model file if present
        fallback = PROJECT_ROOT / "my_waste_model_6_class.pt"
        if fallback.exists():
            model_file = fallback
        else:
            raise FileNotFoundError(
                f"Model weights not found at '{model_file}'.\n"
                f"Please ensure 'my_waste_model_6_class.pt' exists in 'weights/' or project root."
            )

    source_path = resolve_path(source, "assets/sample_images/047.jpg")
    if not source_path.exists():
        raise FileNotFoundError(f"Input source not found at '{source_path}'")

    print(f"\n[INFO] Loading YOLOv8 model from: {model_file}")
    model = YOLO(str(model_file))

    print(f"[INFO] Running inference on: {source_path} (Confidence threshold: {conf})")
    predict_kwargs = {"source": str(source_path), "conf": conf}
    if device is not None:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    all_summary_results = []

    out_directory = None
    if save_dir:
        out_directory = resolve_path(save_dir, "output")
        out_directory.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        image_name = Path(result.path).name if hasattr(result, "path") else f"result_{i}.jpg"
        print(f"\n=========================================")
        print(f"Results for: {image_name}")
        print(f"=========================================")

        detected_items = []
        class_names = result.names

        if not result.boxes or len(result.boxes) == 0:
            print("No waste objects detected in this image.")
        else:
            print("Detected Items:")
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

                print(
                    f"  -> Found: {label:<10} | "
                    f"Category: {category:<4} | "
                    f"Confidence: {confidence:.2f} | "
                    f"BBox: {box_coords}"
                )

        stats = compute_waste_statistics(detected_items)
        report = format_statistics_report(stats)
        print("\n" + report)

        # Handle image saving (headless safe)
        if out_directory:
            save_path = out_directory / f"annotated_{image_name}"
            result.save(filename=str(save_path))
            print(f"[INFO] Annotated visualization saved to: {save_path}")

        # Handle interactive display window with fallback for headless setups
        if show:
            try:
                result.show()
            except Exception as e:
                print(
                    f"[WARNING] Could not open GUI display window ({e}).\n"
                    f"Running in a headless or remote environment. Use --save-dir to inspect output.",
                    file=sys.stderr
                )

        all_summary_results.append({
            "image": image_name,
            "detected_items": detected_items,
            "statistics": stats
        })

    return all_summary_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Waste Detection & Dry/Wet Segregation Analysis"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="assets/sample_images/047.jpg",
        help="Path to input image, directory of images, or video stream (default: assets/sample_images/047.jpg)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/my_waste_model_6_class.pt",
        help="Path to trained YOLOv8 model weights (.pt) (default: weights/my_waste_model_6_class.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output",
        help="Directory to save annotated images (default: output). Set to empty string to disable."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display image in an interactive desktop GUI window (requires graphical desktop)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', '0', 'cuda', etc. (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir if args.save_dir else None

    try:
        predict_waste(
            model_path=args.weights,
            source=args.source,
            conf=args.conf,
            save_dir=save_dir,
            show=args.show,
            device=args.device
        )
    except FileNotFoundError as err:
        print(f"\n[ERROR] File error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR] An unexpected error occurred: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

