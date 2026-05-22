import json
import os
from pathlib import Path
import cv2
import importlib.util
import traceback
import numpy as np

from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_bounding_boxes(state):
    predictor = load_predictor(state)

    if predictor is None:
        state["evaluation"] = {"metric_name": "AP50", "metric_value": 0.0, "status": "predictor_load_failed"}

        return state

    splits = ["val", "test"]

    state["evaluation"] = {}
    state["evaluation_visualizations"] = {}

    for split in splits:

        all_preds = []
        all_gts = {}
        vis_paths = []

        tp_all = []
        fp_all = []
        fn_all = []

        path_to_images = Path(state["split_paths"][split]["images"])
        path_to_labels = Path(state["split_paths"][split]["labels"])

        image_files = list(path_to_images.iterdir())

        # -------------------------
        # 1. COLLECT DATA
        # -------------------------
        for img_path in tqdm(image_files, desc="Evaluating"):
            if not img_path.is_file():
                continue

            try:
                pred_boxes = predictor.predict(str(img_path))
            except Exception:
                logger.error(f"Prediction failed:\n{traceback.format_exc()}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gt_boxes = load_yolo_labels(path_to_labels, img_path.stem, img.shape[:2])

            all_gts[img_path.name] = gt_boxes

            for p in pred_boxes:
                all_preds.append({
                    "image": img_path.name,
                    "bbox": p["bbox"],
                    "score": p.get("confidence", 1.0),
                    "label": p.get("label", None)
                })

        # -------------------------
        # 2. SORT BY CONFIDENCE
        # -------------------------
        all_preds = sorted(all_preds, key=lambda x: x["score"], reverse=True)

        # -------------------------
        # 3. GLOBAL MATCHING (AP CORE)
        # -------------------------
        used_gts = {img: [False] * len(gts) for img, gts in all_gts.items()}

        for pred in all_preds:
            img_name = pred["image"]
            gts = all_gts.get(img_name, [])

            best_iou = 0.0
            best_idx = -1

            for i, gt in enumerate(gts):
                if used_gts[img_name][i]:
                    continue

                iou_val = compute_iou(pred, gt)

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            if best_iou >= 0.5:
                tp_all.append({
                    "image": img_name,
                    "bbox": pred["bbox"],
                    "gt": gts[best_idx]["bbox"],
                    "score": pred["score"]
                })
                used_gts[img_name][best_idx] = True
            else:
                fp_all.append({
                    "image": img_name,
                    "bbox": pred["bbox"],
                    "score": pred["score"]
                })

        # -------------------------
        # 4. FN (MISSED GTS)
        # -------------------------
        for img_name, gts in all_gts.items():
            for i, gt in enumerate(gts):
                if not used_gts[img_name][i]:
                    fn_all.append({
                        "image": img_name,
                        "bbox": gt["bbox"]
                    })

        # -------------------------
        # 5. AP50 (PR CURVE)
        # -------------------------
        pred_scores = []

        for p in tp_all:
            pred_scores.append((p["score"], 1))  # TP

        for p in fp_all:
            pred_scores.append((p["score"], 0))  # FP

        # 2. sort by confidence
        pred_scores = sorted(pred_scores, key=lambda x: x[0], reverse=True)

        if len(pred_scores) == 0:
            ap50 = 0.0
        else:
            tp = np.array([p[1] for p in pred_scores])
            fp = np.array([1 - p[1] for p in pred_scores])

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            recall = tp_cum / (sum(len(v) for v in all_gts.values()) + 1e-9)

            recall_points = np.linspace(0, 1, 101)

            precision_interp = np.interp(recall_points, recall, precision)

            ap50 = float(np.mean(precision_interp))

        # -------------------------
        # 6. VISUALIZATION PER IMAGE
        # -------------------------
        vis_dir = Path(f"workspace/stage_{state.get('stage_id', 0)}_step_{state.get('step_id', 0)}/evaluation/visualizations/{split}/")
        vis_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            vis = visualize_detections(img_path, tp_all, fp_all, fn_all)
            if vis is not None:
                out_path = vis_dir / f"{img_path.stem}_tp_fp_fn.jpg"
                if vis is not None:
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    vis_paths.append(str(out_path))

        # -------------------------
        # 7. SAVE RESULT
        # -------------------------
        state["evaluation"][split] = {
            "metric_name": "AP50",
            "metric_value": round(ap50, 4),
            "status": "success",
            "tp": len(tp_all),
            "fp": len(fp_all),
            "fn": len(fn_all)
        }

        metrics_dir = Path(f"workspace/stage_{state.get('stage_id', 0)}_step_{state.get('step_id', 0)}/evaluation/metrics/{split}/")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        json.dump(state["evaluation"][split], open(metrics_dir / f"{split}_metrics.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        state["evaluation_visualizations"][split] = vis_paths

    return state


def placeholder_evaluation(state, output_type):
    state["evaluation"] = {"status": "not_implemented", "type": output_type}
    return state


def load_yolo_labels(labels_dir, img_stem, img_size):
    """
    labels_dir: Path do labels/val
    img_stem: np. '101_DSC01167'
    img_size: (h, w)
    """

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

            # YOLO normalized -> pixel xyxy
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h

            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "label": int(cls)
            })

    return boxes


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


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"] if isinstance(a, dict) else a
    bx1, by1, bx2, by2 = b["bbox"] if isinstance(b, dict) else b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    return inter / (area_a + area_b - inter + 1e-9)


def visualize_detections(image_path, tp_all, fp_all, fn_all):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    name = Path(image_path).name

    def draw(box, color, text):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for d in tp_all:
        if d["image"] == name:
            draw(d["bbox"], (0, 255, 0), "TP")

    for d in fp_all:
        if d["image"] == name:
            draw(d["bbox"], (255, 0, 0), "FP")

    for d in fn_all:
        if d["image"] == name:
            draw(d["bbox"], (255, 255, 0), "FN")

    return img
