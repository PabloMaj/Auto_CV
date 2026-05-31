import numpy as np


def compute_ap50(tp_all, fp_all, total_gt):

    pred_scores = []

    for tp in tp_all:
        pred_scores.append((tp["score"], 1))

    for fp in fp_all:
        pred_scores.append((fp["score"], 0))

    pred_scores = sorted(
        pred_scores,
        key=lambda x: x[0],
        reverse=True
    )

    if len(pred_scores) == 0:
        return 0.0

    tp = np.array([x[1] for x in pred_scores])
    fp = np.array([1 - x[1] for x in pred_scores])

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / (total_gt + 1e-9)

    recall_points = np.linspace(0, 1, 101)

    precision_interp = np.interp(
        recall_points,
        recall,
        precision
    )

    return float(np.mean(precision_interp))
