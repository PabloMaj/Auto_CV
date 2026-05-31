import os
from pathlib import Path
import importlib.util
import traceback
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_yolo_boxes(labels_dir, img_stem, img_size):
    labels_dir = Path(labels_dir)
    label_file = labels_dir / f"{img_stem}.txt"

    if not label_file.exists():
        return []

    h, w = img_size

    boxes = []

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, x, y, bw, bh = map(float, parts)

            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h

            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "label": int(cls)
            })

    return boxes


def load_yolo_points(labels_dir, img_stem, img_shape=None):
    """
    Format:
    class_id x y  (normalized)
    """

    h, w = img_shape

    labels_dir = Path(labels_dir)
    label_file = labels_dir / f"{img_stem}.txt"

    if not label_file.exists():
        return []

    points = []

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 3:
                continue

            cls, x, y = map(float, parts)

            x = x * w
            y = y * h

            points.append({
                "point": [x, y],
                "label": int(cls)
            })

    return points


def load_yolo_lines(labels_dir, img_stem, img_shape=None):
    """
    Format:
    class_id x1 y1 x2 y2 (normalized)
    """

    h, w = img_shape

    labels_dir = Path(labels_dir)
    label_file = labels_dir / f"{img_stem}.txt"

    if not label_file.exists():
        return []

    lines = []

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, x1, y1, x2, y2 = map(float, parts)

            x1 *= w
            y1 *= h
            x2 *= w
            y2 *= h

            lines.append({
                "line": [x1, y1, x2, y2],
                "label": int(cls)
            })

    return lines


def load_predictor(state):
    stage_id = state.get("stage_id")
    step_id = state.get("step_id")

    PREDICTOR_TEMPLATE = r"C:\projects\agent_cv\workspace\stage_{stage_id}_step_{step_id}\generated_solution.py"

    path = PREDICTOR_TEMPLATE.format(stage_id=stage_id, step_id=step_id)

    if not os.path.exists(path):
        logger.error(f"Missing predictor: {path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("generated_solution", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.Predictor()

    except Exception:
        logger.error(traceback.format_exc())
        return None
