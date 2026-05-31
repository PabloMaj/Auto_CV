import numpy as np


def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_point_metrics(tp_all, fp_all, fn_all, all_gts):

    tp = len(tp_all)
    fp = len(fp_all)
    fn = len(fn_all)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # mean distance for TP
    if tp > 0:
        mean_dist = float(
            np.mean([
                _dist(t["point"], t["gt"])
                for t in tp_all
            ])
        )
    else:
        mean_dist = float("inf")

    return {
        "metric_name": "F1",
        "metric_value": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_distance": round(mean_dist, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "status": "success"
    }
