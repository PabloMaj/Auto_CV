"""
Test-case generation for LLM-judge calibration.

For each variant (bboxes / midpoints / lines) we generate test cases with
ground-truth F1 values spread across [0.4, 1.0].

Strategy
--------
1. Collect all (image, label) pairs from the val split of every dataset.
2. For each test case sample a random image, load its GT annotations, and
   create a perturbed prediction set by:
     - keeping a random fraction of GT annotations (with small jitter → TP)
     - dropping the rest               → FN
     - adding random fake detections  → FP
3. Run the real matcher to compute the actual F1 (ground truth for the experiment).
4. Accept the case only when F1 falls in the target bin [bin_lo, bin_hi).
5. Select 20 cases covering the full range [0.4, 1.0] (5 per bin × 4 bins).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.funcs.evaluator_funcs.matchers.box_matcher import BoxMatcher
from src.funcs.evaluator_funcs.matchers.point_matcher import PointMatcher
from src.funcs.evaluator_funcs.matchers.line_matcher import LineMatcher
from src.funcs.evaluator_funcs.utils.loaders import (
    load_yolo_boxes,
    load_yolo_points,
    load_yolo_lines,
)

# Match parameters must mirror the production evaluators exactly.
_BOX_MATCHER = BoxMatcher(iou_threshold=0.5)
_POINT_MATCHER = PointMatcher(distance_threshold=25)
_LINE_MATCHER = LineMatcher(lateral_threshold=20, angle_threshold=2.5, overlap_threshold=0.5)

CYAN_BGR = (255, 255, 0)  # cyan in OpenCV BGR


@dataclass
class TestCase:
    image_path: Path
    dataset: str
    variant: str
    predictions: list          # list of raw coord lists, format depends on variant
    gt_annotations: list       # list of dicts from loader
    actual_f1: float
    vis_bgr: np.ndarray = field(repr=False)  # BGR image with CYAN predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gt(label_dir: Path, stem: str, img_shape, variant: str) -> list:
    h, w = img_shape
    if variant == "bboxes":
        return load_yolo_boxes(label_dir, stem, (h, w))
    if variant == "midpoints":
        return load_yolo_points(label_dir, stem, (h, w))
    return load_yolo_lines(label_dir, stem, (h, w))


def _compute_f1(predictions: list, gt_annotations: list, img_name: str, variant: str) -> float:
    """Run the production matcher and return F1 for a single image."""
    img_key = img_name
    gts = {img_key: gt_annotations}

    if variant == "bboxes":
        preds = [{"image": img_key, "bbox": p, "score": 1.0} for p in predictions]
        tp, fp, fn = _BOX_MATCHER.match(preds, gts)
    elif variant == "midpoints":
        preds = [{"image": img_key, "point": p, "score": 1.0} for p in predictions]
        tp, fp, fn = _POINT_MATCHER.match(preds, gts)
    else:
        preds = [{"image": img_key, "line": p, "score": 1.0} for p in predictions]
        tp, fp, fn = _LINE_MATCHER.match(preds, gts)

    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    return 2 * n_tp / (2 * n_tp + n_fp + n_fn + 1e-9)


def _perturb(gt_annotations: list, variant: str, img_shape, tp_count: int,
             fp_count: int, rng: np.random.Generator) -> list:
    """Return a perturbed prediction list with approximately tp_count TPs and fp_count FPs."""
    h, w = img_shape
    n = len(gt_annotations)
    tp_count = max(0, min(tp_count, n))

    indices = rng.choice(n, tp_count, replace=False) if tp_count > 0 else []
    predictions = []

    for i in indices:
        ann = gt_annotations[i]
        if variant == "bboxes":
            x1, y1, x2, y2 = ann["bbox"]
            bw, bh = x2 - x1, y2 - y1
            sigma = max(1.0, min(bw, bh) * 0.04)
            j = rng.normal(0, sigma, 4)
            predictions.append([
                float(np.clip(x1 + j[0], 0, w)),
                float(np.clip(y1 + j[1], 0, h)),
                float(np.clip(x2 + j[2], 0, w)),
                float(np.clip(y2 + j[3], 0, h)),
            ])
        elif variant == "midpoints":
            x, y = ann["point"]
            j = rng.normal(0, 4, 2)
            predictions.append([
                float(np.clip(x + j[0], 0, w)),
                float(np.clip(y + j[1], 0, h)),
            ])
        else:
            x1, y1, x2, y2 = ann["line"]
            j = rng.normal(0, 4, 4)
            predictions.append([
                float(np.clip(x1 + j[0], 0, w)),
                float(np.clip(y1 + j[1], 0, h)),
                float(np.clip(x2 + j[2], 0, w)),
                float(np.clip(y2 + j[3], 0, h)),
            ])

    # FP predictions
    for _ in range(fp_count):
        if variant == "bboxes":
            bw = float(rng.uniform(15, 70))
            bh = float(rng.uniform(15, 70))
            x1 = float(rng.uniform(0, max(1, w - bw)))
            y1 = float(rng.uniform(0, max(1, h - bh)))
            predictions.append([x1, y1, x1 + bw, y1 + bh])
        elif variant == "midpoints":
            predictions.append([
                float(rng.uniform(0, w)),
                float(rng.uniform(0, h)),
            ])
        else:
            # FP lines: random position, random angle (clearly off-pattern)
            x1 = float(rng.uniform(0.1 * w, 0.9 * w))
            y1 = float(rng.uniform(0.1 * h, 0.9 * h))
            angle = float(rng.uniform(0, np.pi))
            length = float(rng.uniform(0.15 * min(w, h), 0.45 * min(w, h)))
            x2 = float(np.clip(x1 + length * np.cos(angle), 0, w))
            y2 = float(np.clip(y1 + length * np.sin(angle), 0, h))
            predictions.append([x1, y1, x2, y2])

    rng.shuffle(predictions)
    return predictions


def _render_vis(image_bgr: np.ndarray, predictions: list, variant: str) -> np.ndarray:
    vis = image_bgr.copy()
    for pred in predictions:
        if variant == "bboxes":
            x1, y1, x2, y2 = map(int, pred)
            cv2.rectangle(vis, (x1, y1), (x2, y2), CYAN_BGR, 2)
        elif variant == "midpoints":
            x, y = map(int, pred)
            cv2.circle(vis, (x, y), 6, CYAN_BGR, -1)
        else:
            x1, y1, x2, y2 = map(int, pred)
            cv2.line(vis, (x1, y1), (x2, y2), CYAN_BGR, 2)
    return vis


def _tp_fp_from_target_f1(n_gt: int, target_f1: float, strategy: int,
                          rng: np.random.Generator):
    """
    Derive (tp_count, fp_count) that should approximately give target_f1.

    strategy 0 — drop GT, no FP  (pure recall reduction)
    strategy 1 — keep all GT, add FP  (pure precision reduction)
    strategy 2 — mixed drop + FP
    """
    f = max(0.05, min(0.999, target_f1))

    if strategy == 0:
        # F1 = 2*TP / (TP + n_gt)  (FP=0, FN=n_gt-TP)
        tp = max(1, round(f * n_gt / (2 - f)))
        fp = 0
    elif strategy == 1:
        # F1 = 2*n_gt / (2*n_gt + FP)
        tp = n_gt
        fp = max(0, round(2 * n_gt * (1 - f) / f))
    else:
        # mixed: random recall
        recall = float(rng.uniform(0.4, 1.0))
        tp = max(1, round(recall * n_gt))
        fn = n_gt - tp
        # solve for FP: F1 = 2*TP/(2*TP + FP + FN)  → FP = (2*TP - f*(TP+FN)) / f
        fp_exact = (2 * tp - f * (tp + fn)) / f
        fp = max(0, round(fp_exact + float(rng.uniform(-1, 1))))

    return tp, fp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_image_label_pairs(data_root: Path, variant: str, split: str = "val") -> list:
    """Return list of (image_path, label_dir, dataset_name) for all datasets.

    Expected structure: {dataset}/images/{split}/ and {dataset}/labels/{split}/
    """
    suffix = {"bboxes": "bboxes", "midpoints": "midpoints", "lines": "lines"}[variant]
    pairs = []
    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.endswith(f"_{suffix}"):
            continue
        img_dir = ds_dir / "images" / split
        lbl_dir = ds_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                pairs.append((img_path, lbl_dir, ds_dir.name))
    return pairs


def generate_test_cases(
    variant: str,
    data_root: str | Path,
    n_cases: int = 20,
    seed: int = 42,
) -> list[TestCase]:
    """
    Generate n_cases TestCase objects with F1 in [0.40, 1.0], evenly distributed
    across 4 bins: [0.40, 0.55), [0.55, 0.70), [0.70, 0.85), [0.85, 1.01).
    """
    data_root = Path(data_root)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    pairs = find_image_label_pairs(data_root, variant)
    if not pairs:
        raise ValueError(f"No image-label pairs found for variant '{variant}' in {data_root}")

    bins = [
        (0.40, 0.55),
        (0.55, 0.70),
        (0.70, 0.85),
        (0.85, 1.01),
    ]
    cases_per_bin = n_cases // len(bins)

    test_cases: list[TestCase] = []

    for bin_lo, bin_hi in bins:
        collected = []
        attempts = 0

        while len(collected) < cases_per_bin and attempts < 300:
            attempts += 1

            img_path, lbl_dir, ds_name = pairs[rng.integers(len(pairs))]

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            gt_anns = _load_gt(lbl_dir, img_path.stem, (h, w), variant)

            if len(gt_anns) < 3:
                continue

            n_gt = len(gt_anns)
            target_f1 = float(rng.uniform(bin_lo, min(bin_hi, 1.0)))
            strategy = int(rng.integers(3))
            tp_count, fp_count = _tp_fp_from_target_f1(n_gt, target_f1, strategy, rng)

            preds = _perturb(gt_anns, variant, (h, w), tp_count, fp_count, rng)
            if not preds:
                continue

            actual_f1 = _compute_f1(preds, gt_anns, img_path.name, variant)

            if not (bin_lo <= actual_f1 < bin_hi):
                continue

            vis = _render_vis(img_bgr, preds, variant)
            collected.append(TestCase(
                image_path=img_path,
                dataset=ds_name,
                variant=variant,
                predictions=preds,
                gt_annotations=gt_anns,
                actual_f1=actual_f1,
                vis_bgr=vis,
            ))

        test_cases.extend(collected)
        if len(collected) < cases_per_bin:
            print(f"  [WARNING] bin [{bin_lo:.2f}, {bin_hi:.2f}): "
                  f"found only {len(collected)}/{cases_per_bin} cases after {attempts} attempts")

    f1_values = [tc.actual_f1 for tc in test_cases]
    print(f"  [{variant}] {len(test_cases)} test cases  "
          f"F1 range [{min(f1_values):.3f}, {max(f1_values):.3f}]  "
          f"mean={np.mean(f1_values):.3f}")
    return test_cases
