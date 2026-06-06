import random
from pathlib import Path

import cv2

from src.funcs.evaluator_funcs.utils.loaders import (
    load_yolo_boxes,
    load_yolo_lines,
    load_yolo_points,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GT_COLOR = (0, 220, 80)   # green (BGR)
_GT_THICKNESS = 2
_MAX_IMAGES = 4
_RESIZE_FACTOR = 0.25


def sample_training_images(state, label_free: bool = False) -> list:
    """
    Sample up to _MAX_IMAGES training images.

    label_free=False  — overlay GT annotations in green before returning paths.
    label_free=True   — return raw images with no annotation.

    Images are resized to 1/4 of original resolution and saved to
    workspace/stage_X_step_Y/_programmer_vis/ for use in the LLM prompt.
    """
    # Use eval_split_paths so annotations match the task output (midpoints/lines/etc.)
    eval_split_paths = state.get("eval_split_paths", state.get("split_paths", {}))
    image_dir = Path(eval_split_paths.get("train", {}).get("images", ""))
    label_dir = Path(eval_split_paths.get("train", {}).get("labels", ""))
    desired_out = state.get("desired_output", "unknown")
    stage_id = state.get("stage_id", 0)
    step_id = state.get("step_id", 0)

    if not image_dir.exists():
        logger.warning(f"Training image dir not found: {image_dir}")
        return []

    image_files = [p for p in image_dir.iterdir() if p.is_file()]
    if not image_files:
        return []

    sampled = random.sample(image_files, min(_MAX_IMAGES, len(image_files)))

    out_dir = Path("workspace") / state.get("exp_id", "default") / f"stage_{stage_id}_step_{step_id}/_programmer_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for img_path in sampled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if not label_free and label_dir.exists():
            img = _overlay_gt(img, label_dir, img_path.stem, desired_out)

        h, w = img.shape[:2]
        small = cv2.resize(
            img,
            (max(1, int(w * _RESIZE_FACTOR)), max(1, int(h * _RESIZE_FACTOR))),
            interpolation=cv2.INTER_AREA,
        )

        out = out_dir / img_path.name
        cv2.imwrite(str(out), small)
        paths.append(str(out))

    logger.info(f"Sampled {len(paths)} training image(s) for programmer prompt "
                f"(label_free={label_free})")
    return paths


# ---------------------------------------------------------------------------
# GT annotation helpers
# ---------------------------------------------------------------------------

def _overlay_gt(img, label_dir: Path, img_stem: str, desired_out: str):
    h, w = img.shape[:2]

    if desired_out == "line_segments":
        for gt in load_yolo_lines(label_dir, img_stem, img_shape=(h, w)):
            x1, y1, x2, y2 = [int(v) for v in gt["line"]]
            cv2.line(img, (x1, y1), (x2, y2), _GT_COLOR, _GT_THICKNESS)

    elif desired_out == "bounding_boxes":
        for gt in load_yolo_boxes(label_dir, img_stem, img_size=(h, w)):
            x1, y1, x2, y2 = [int(v) for v in gt["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), _GT_COLOR, _GT_THICKNESS)

    elif desired_out in ("points", "midpoints"):
        for gt in load_yolo_points(label_dir, img_stem, img_shape=(h, w)):
            x, y = int(gt["point"][0]), int(gt["point"][1])
            cv2.circle(img, (x, y), 5, _GT_COLOR, -1)

    return img
