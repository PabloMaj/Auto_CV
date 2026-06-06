import numpy as np


def compute_ap50(tp_all, fp_all, total_gt):
    """
    101-point interpolated AP@IoU=0.5.

    Uses max-precision-to-the-right at each recall threshold so recall
    regions that were never achieved contribute 0 (not the last precision value).
    """
    pred_scores = (
        [(tp["score"], 1) for tp in tp_all]
        + [(fp["score"], 0) for fp in fp_all]
    )

    if not pred_scores or total_gt == 0:
        return 0.0

    pred_scores.sort(key=lambda x: x[0], reverse=True)

    tp_arr = np.array([x[1] for x in pred_scores], dtype=float)
    fp_arr = 1.0 - tp_arr

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / (total_gt + 1e-9)

    # 101-point: at each threshold take max precision where recall >= threshold
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        mask = recall >= thr
        ap += float(np.max(precision[mask])) if mask.any() else 0.0

    return ap / 101
