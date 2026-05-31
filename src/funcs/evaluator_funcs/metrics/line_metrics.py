import numpy as np


def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_line_metrics(tp_all, fp_all, fn_all, all_gts):

    tp = len(tp_all)
    fp = len(fp_all)
    fn = len(fn_all)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # -----------------------------
    # GEOMETRIC ERRORS (TP only)
    # -----------------------------
    if tp > 0:

        endpoint_error = np.mean([
            (_dist(t["line"][:2], t["gt"][:2]) + _dist(t["line"][2:], t["gt"][2:])) / 2
            for t in tp_all
        ])

        # optional: angle error if stored by matcher
        angle_error = np.mean([
            t.get("angle_error", 0.0)
            for t in tp_all
        ])

    else:
        endpoint_error = float("inf")
        angle_error = float("inf")

    return {
        "metric_name": "LINE_F1",
        "metric_value": round(f1, 4),

        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),

        "endpoint_error": float(endpoint_error),
        "angle_error": float(angle_error),

        "tp": tp,
        "fp": fp,
        "fn": fn,

        "status": "success"
    }
