"""
Model Training Module for YOLOv8 Waste Classification & Detection.
Handles transfer learning fine-tuning from pretrained YOLOv8 checkpoints.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

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
from src.utils import PROJECT_ROOT, resolve_path


def train_model(
    data_config: str = "data/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 40,
    batch_size: int = 8,
    imgsz: int = 640,
    device: Optional[str] = None,
    output_model_name: str = "my_waste_model_6_class.pt",
    weights_dir: str = "weights"
) -> Path:
    """
    Trains/fine-tunes YOLOv8 on the waste dataset.

    Args:
        data_config: Path to data.yaml
        base_model: Pretrained YOLOv8 weights (e.g. yolov8n.pt)
        epochs: Number of training epochs
        batch_size: Training batch size
        imgsz: Image resolution
        device: 'cpu', 'cuda', '0', etc.
        output_model_name: File name for the final trained checkpoint
        weights_dir: Target directory to store output model

    Returns:
        Path to the saved best model.
    """
    data_path = resolve_path(data_config, "data/data.yaml")
    if not data_path.exists():
        # Fallback to root data.yaml
        fallback_data = PROJECT_ROOT / "data.yaml"
        if fallback_data.exists():
            data_path = fallback_data
        else:
            raise FileNotFoundError(
                f"Dataset configuration file not found at '{data_path}'.\n"
                f"Please ensure 'data.yaml' exists in 'data/' or project root."
            )

    print(f"\n=========================================")
    print("      STARTING YOLOV8 TRAINING           ")
    print(f"=========================================")
    print(f"Dataset Config : {data_path}")
    print(f"Base Weights   : {base_model}")
    print(f"Epochs         : {epochs}")
    print(f"Batch Size     : {batch_size}")
    print(f"Image Size     : {imgsz}")
    print(f"Device         : {'Auto' if device is None else device}")
    print(f"=========================================\n")

    model = YOLO(base_model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
    }
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    print("\n--- Training Finished Successfully ---")

    best_checkpoint = Path(results.save_dir) / "weights" / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = Path(results.save_dir) / "weights" / "last.pt"

    target_dir = resolve_path(weights_dir, "weights")
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / output_model_name

    # Safely copy trained weights to destination
    if best_checkpoint.exists():
        shutil.copy2(best_checkpoint, destination)
        print(f"[SUCCESS] Trained model saved to: {destination}")
    else:
        print(f"[WARNING] Checkpoint file not found at: {best_checkpoint}")
        destination = best_checkpoint

    return destination


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on Waste Classification Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to dataset configuration YAML (default: data/data.yaml)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO base model or checkpoint (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs (default: 40)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size resolution in pixels (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target compute device: 'cpu', '0', etc. (default: auto)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="my_waste_model_6_class.pt",
        help="Output checkpoint filename (default: my_waste_model_6_class.pt)"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Destination directory for trained weights (default: weights/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_model(
            data_config=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            output_model_name=args.output_name,
            weights_dir=args.weights_dir
        )
    except FileNotFoundError as err:
        print(f"\n[ERROR] Configuration error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR] Training failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

