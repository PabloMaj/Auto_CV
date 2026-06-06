import numpy as np


class LineMatcher:

    def __init__(self, lateral_threshold=25, angle_threshold=10, overlap_threshold=0.3):
        self.lateral_threshold = lateral_threshold
        self.angle_threshold = angle_threshold
        self.overlap_threshold = overlap_threshold

    def _angle(self, l1, l2):

        v1 = np.array(l1[2:], dtype=np.float64) - np.array(l1[:2], dtype=np.float64)
        v2 = np.array(l2[2:], dtype=np.float64) - np.array(l2[:2], dtype=np.float64)

        v1 /= np.linalg.norm(v1) + 1e-9
        v2 /= np.linalg.norm(v2) + 1e-9

        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))

        return min(angle, 180 - angle)

    def _point_line_distance(self, p, a, b):

        p = np.asarray(p)
        a = np.asarray(a)
        b = np.asarray(b)

        return abs(np.cross(b - a, p - a)) / (np.linalg.norm(b - a) + 1e-9)

    def _sample_points(self, line, n=4):
        p1 = np.array(line[:2], dtype=np.float64)
        p2 = np.array(line[2:], dtype=np.float64)
        return [p1 + t * (p2 - p1) for t in np.linspace(0, 1, n)]

    def _mean_lateral_distance(self, pred_line, gt_line):
        points = self._sample_points(gt_line, n=4)
        a = np.array(pred_line[:2], dtype=np.float64)
        b = np.array(pred_line[2:], dtype=np.float64)
        return float(np.mean([self._point_line_distance(p, a, b) for p in points]))

    def _overlap_ratio(self, l1, l2):

        p1 = np.array(l1[:2])
        p2 = np.array(l1[2:])

        direction = np.array(p2 - p1, dtype=np.float64)
        direction /= np.linalg.norm(direction) + 1e-9

        gt_proj = [
            np.dot(np.array(l2[:2]) - p1, direction),
            np.dot(np.array(l2[2:]) - p1, direction)
        ]

        pr_proj = [
            np.dot(np.array(l1[:2]) - p1, direction),
            np.dot(np.array(l1[2:]) - p1, direction)
        ]

        gt_min, gt_max = sorted(gt_proj)
        pr_min, pr_max = sorted(pr_proj)

        overlap = max(0, min(gt_max, pr_max) - max(gt_min, pr_min))

        gt_length = gt_max - gt_min

        return overlap / (gt_length + 1e-9)

    def match(self, predictions, ground_truths):

        tp_all, fp_all, fn_all = [], [], []

        used_gts = {
            img: [False] * len(gts)
            for img, gts in ground_truths.items()
        }

        for pred in predictions:

            img = pred["image"]
            gts = ground_truths.get(img, [])

            best_idx = -1
            best_score = np.inf
            best_angle = None

            for i, gt in enumerate(gts):

                if used_gts[img][i]:
                    continue

                angle = self._angle(pred["line"], gt["line"])

                if angle > self.angle_threshold:
                    continue

                lateral_distance = self._mean_lateral_distance(
                    pred["line"], gt["line"]
                )

                overlap = self._overlap_ratio(
                    pred["line"],
                    gt["line"]
                )

                if overlap < self.overlap_threshold:
                    continue

                if lateral_distance > self.lateral_threshold:
                    continue

                if lateral_distance < best_score:
                    best_score = lateral_distance
                    best_idx = i
                    best_angle = angle

            if best_idx >= 0:

                tp_all.append({
                    "image": img,
                    "line": pred["line"],
                    "gt": gts[best_idx]["line"],
                    "score": pred.get("score", None),
                    "angle_error": best_angle
                })

                used_gts[img][best_idx] = True

            else:

                fp_all.append({
                    "image": img,
                    "line": pred["line"],
                    "score": pred.get("score", None)
                })

        for img, gts in ground_truths.items():
            for i, gt in enumerate(gts):

                if not used_gts[img][i]:
                    fn_all.append({
                        "image": img,
                        "line": gt["line"]
                    })

        return tp_all, fp_all, fn_all
