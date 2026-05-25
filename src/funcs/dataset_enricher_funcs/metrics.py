import numpy as np
from scipy.optimize import linear_sum_assignment


class DetectionAP50:
    def evaluate_image(self, gt_boxes, pred_boxes):
        if len(gt_boxes) == 0:
            return 0.0

        if len(pred_boxes) == 0:
            return 0.0

        iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))

        for i, g in enumerate(gt_boxes):
            for j, p in enumerate(pred_boxes):
                iou_mat[i, j] = bbox_iou(g, p)

        cost = 1.0 - iou_mat
        r, c = linear_sum_assignment(cost)

        tp = (iou_mat[r, c] >= 0.5).sum()
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        return precision * recall  # AP@0.50 (uprośćmy)

    def evaluate_dataset(self, gt_list, pred_list):
        scores = [
            self.evaluate_image(gt, pr)
            for gt, pr in zip(gt_list, pred_list)
        ]
        return float(np.mean(scores))


class SegmentationAP50:
    def evaluate_image(self, gt_masks, pred_masks):
        if len(gt_masks) == 0:
            return 0.0

        if len(pred_masks) == 0:
            return 0.0

        iou_mat = np.zeros((len(gt_masks), len(pred_masks)))

        for i, g in enumerate(gt_masks):
            for j, p in enumerate(pred_masks):
                iou_mat[i, j] = mask_iou(g, p)

        cost = 1.0 - iou_mat
        r, c = linear_sum_assignment(cost)

        tp = (iou_mat[r, c] >= 0.5).sum()
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        return precision * recall

    def evaluate_dataset(self, gt_list, pred_list):
        scores = [
            self.evaluate_image(gt, pr)
            for gt, pr in zip(gt_list, pred_list)
        ]
        return float(np.mean(scores))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same shape")

    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)
