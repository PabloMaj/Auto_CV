import numpy as np


class LineMatcher:

    def __init__(self, distance_threshold=100, angle_threshold=10.0):
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold

    def _dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _angle(self, l1, l2):

        v1 = np.array(l1[2:]) - np.array(l1[:2])
        v2 = np.array(l2[2:]) - np.array(l2[:2])

        norm1 = np.linalg.norm(v1) + 1e-9
        norm2 = np.linalg.norm(v2) + 1e-9

        cos = np.dot(v1, v2) / (norm1 * norm2)
        cos = np.clip(cos, -1.0, 1.0)

        return np.degrees(np.arccos(cos))

    def match(self, predictions, ground_truths):

        tp_all, fp_all, fn_all = [], [], []

        used_gts = {
            img: [False] * len(gts)
            for img, gts in ground_truths.items()
        }

        for pred in predictions:

            img = pred["image"]
            gts = ground_truths.get(img, [])

            best_score = float("inf")
            best_idx = -1

            for i, gt in enumerate(gts):

                if used_gts[img][i]:
                    continue

                # endpoint distance
                d1 = self._dist(pred["line"][:2], gt["line"][:2])
                d2 = self._dist(pred["line"][2:], gt["line"][2:])
                dist = (d1 + d2) / 2

                # angle difference
                angle = self._angle(pred["line"], gt["line"])

                if dist < best_score:
                    best_score = dist
                    best_idx = i
                    best_angle = angle

            # FINAL DECISION: BOTH CONDITIONS
            if (best_score <= self.distance_threshold) and (best_angle <= self.angle_threshold):

                tp_all.append({
                    "image": img,
                    "line": pred["line"],
                    "gt": gts[best_idx]["line"],
                    "score": pred["score"],
                    "angle_error": best_angle
                })

                used_gts[img][best_idx] = True

            else:

                fp_all.append({
                    "image": img,
                    "line": pred["line"],
                    "score": pred["score"]
                })

        # FN
        for img, gts in ground_truths.items():
            for i, gt in enumerate(gts):
                if not used_gts[img][i]:
                    fn_all.append({
                        "image": img,
                        "line": gt["line"]
                    })

        return tp_all, fp_all, fn_all
