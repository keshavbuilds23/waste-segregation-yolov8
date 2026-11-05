"""
Utility functions and configuration for the Waste Segregation YOLOv8 project.
Handles waste stream mapping (Dry vs. Wet), statistics calculation, and path resolution.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 6 Detection Classes defined in dataset
WASTE_CLASSES: List[str] = [
    "cardboard",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

# High-level waste stream mapping
# Organic waste -> WET waste
# Cardboard, glass, metal, paper, plastic -> DRY waste
WET_CLASSES: set = {"organic"}
DRY_CLASSES: set = {"cardboard", "glass", "metal", "paper", "plastic"}


def get_waste_category(class_name: str) -> str:
    """
    Maps an individual waste material class to a high-level waste stream category.

    Args:
        class_name: Name of detected class (e.g. 'organic', 'plastic')

    Returns:
        'WET' if organic, otherwise 'DRY'.
    """
    lower_name = class_name.lower().strip()
    if lower_name in WET_CLASSES:
        return "WET"
    return "DRY"


def compute_waste_statistics(detected_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary counts, percentages, and breakdowns from detection results.

    Args:
        detected_items: List of dicts, each with keys 'label', 'confidence', 'category', 'box'.

    Returns:
        Dictionary containing counts and percentages of Dry and Wet waste.
    """
    dry_count = 0
    wet_count = 0
    class_counts: Dict[str, int] = {cls_name: 0 for cls_name in WASTE_CLASSES}

    for item in detected_items:
        label = item.get("label", "").lower()
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

        category = item.get("category", get_waste_category(label))
        if category == "WET":
            wet_count += 1
        else:
            dry_count += 1

    total_items = dry_count + wet_count
    dry_percent = (dry_count / total_items * 100.0) if total_items > 0 else 0.0
    wet_percent = (wet_count / total_items * 100.0) if total_items > 0 else 0.0

    return {
        "total_items": total_items,
        "dry_count": dry_count,
        "wet_count": wet_count,
        "dry_percent": dry_percent,
        "wet_percent": wet_percent,
        "class_counts": class_counts
    }


def format_statistics_report(stats: Dict[str, Any]) -> str:
    """
    Formats the waste segregation statistics into a clean, human-readable report.
    """
    total = stats["total_items"]
    lines = [
        "=========================================",
        "       WASTE SEGREGATION REPORT          ",
        "=========================================",
        f"Total items detected : {total}",
        f"  - Dry Waste items  : {stats['dry_count']}",
        f"  - Wet Waste items  : {stats['wet_count']}",
        "-----------------------------------------",
    ]

    if total > 0:
        lines.append(f"  Dry Waste Composition : {stats['dry_percent']:.2f}%")
        lines.append(f"  Wet Waste Composition : {stats['wet_percent']:.2f}%")
    else:
        lines.append("No dry or wet waste items detected.")

    lines.append("-----------------------------------------")
    lines.append("Class Breakdown:")
    for cls_name, count in stats["class_counts"].items():
        if count > 0:
            category = get_waste_category(cls_name)
            lines.append(f"  * {cls_name:<10} : {count:3d} items ({category})")
    lines.append("=========================================")

    return "\n".join(lines)


def resolve_path(path_str: Optional[str], default_relative: str) -> Path:
    """
    Resolves a file path either as absolute or relative to the project root.
    """
    if path_str:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p
    return PROJECT_ROOT / default_relative

