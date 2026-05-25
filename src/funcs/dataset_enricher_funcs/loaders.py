import cv2
import numpy as np
from pathlib import Path


def load_yolo_bboxes(label_path, img_shape):
    """
    Supports:
    - YOLO detection: class cx cy w h
    - YOLO instance segmentation: class x1 y1 x2 y2 ... xn yn
    Returns list of (x1, y1, x2, y2)
    """
    h, w = img_shape
    boxes = []

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.split()))
            coords = parts[1:]

            # --- klasyczny YOLO bbox ---
            if len(coords) == 4:
                cx, cy, bw, bh = coords
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((x1, y1, x2, y2))

            # --- segmentacja instancyjna → bbox z punktów ---
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs = []
                ys = []
                for i in range(0, len(coords), 2):
                    xs.append(int(coords[i] * w))
                    ys.append(int(coords[i + 1] * h))

                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                boxes.append((x1, y1, x2, y2))

    return boxes


def load_yolo_segment_masks(
    label_path: Path,
    img_shape: tuple[int, int],
) -> list[np.ndarray]:
    """
    Loads YOLO instance segmentation labels and returns binary masks.

    Args:
        label_path : path to YOLO-seg .txt
        img_shape  : (H, W)

    Returns:
        List[np.ndarray] of shape (H, W), dtype=bool
    """
    h, w = img_shape
    masks = []

    if not label_path.exists():
        return masks

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.split()))
            coords = parts[1:]

            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            polygon = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                polygon.append([x, y])

            polygon = np.array(polygon, dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)

            masks.append(mask.astype(bool))

    return masks


def load_images_and_labels(
    image_dir: Path,
    label_dir: Path,
    img_exts=(".png", ".jpg", ".jpeg"),
):
    images = []
    for ext in img_exts:
        images.extend(image_dir.glob(f"*{ext}"))

    images = sorted(images)
    labels = [label_dir / f"{p.stem}.txt" for p in images]

    assert all(label.exists() for label in labels), "Missing label files"
    return images, labels
