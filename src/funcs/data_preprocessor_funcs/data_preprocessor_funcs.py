"""Helper functions for DataPreprocessorAgent."""

from pathlib import Path


SPLITS = ["train", "val", "test", "unlabelled"]


# =========================
# SPLIT PATHS
# =========================
def get_split_paths(dataset_path):
    dataset_path = Path(dataset_path)

    return {
        split: {
            "images": dataset_path / "images" / split,
            "labels": dataset_path / "labels" / split if split != "unlabelled" else None
        }
        for split in SPLITS
    }


# =========================
# COUNT IMAGES
# =========================
def count_images(split_paths):
    return {
        split: len(list(paths["images"].glob("*")))
        for split, paths in split_paths.items()
        if paths["images"].exists()
    }


# =========================
# COUNT OBJECTS (IMPORTANT FIX)
# =========================
def count_objects(split_paths):
    """
    Counts total number of bounding boxes across all YOLO label files.
    """
    result = {}

    for split, paths in split_paths.items():
        if split == "unlabelled":
            result[split] = 0
            continue

        if paths["labels"] is None or not paths["labels"].exists():
            result[split] = 0
            continue

        total_objects = 0

        for label_file in paths["labels"].glob("*.txt"):
            try:
                lines = label_file.read_text().strip().split("\n")
                lines = [l_ for l_ in lines if l_.strip()]
                total_objects += len(lines)
            except Exception:
                continue

        result[split] = total_objects

    return result


# =========================
# VIS PATH
# =========================
def get_vis_path(dataset_path):
    return Path(dataset_path) / "images" / "vis_temp"
