import numpy as np


class PointMatcher:

    def __init__(self, distance_threshold=25):
        self.distance_threshold = distance_threshold

    def _dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def match(self, predictions, ground_truths):

        tp_all, fp_all, fn_all = [], [], []

        used_gts = {
            img: [False] * len(gts)
            for img, gts in ground_truths.items()
        }

        for pred in predictions:

            img = pred["image"]
            gts = ground_truths.get(img, [])

            best_dist = float("inf")
            best_idx = -1

            for i, gt in enumerate(gts):

                if used_gts[img][i]:
                    continue

                dist = self._dist(pred["point"], gt["point"])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_dist <= self.distance_threshold:

                tp_all.append({
                    "image": img,
                    "point": pred["point"],
                    "gt": gts[best_idx]["point"],
                    "score": pred["score"]
                })

                used_gts[img][best_idx] = True

            else:

                fp_all.append({
                    "image": img,
                    "point": pred["point"],
                    "score": pred["score"]
                })

        for img, gts in ground_truths.items():
            for i, gt in enumerate(gts):
                if not used_gts[img][i]:
                    fn_all.append({
                        "image": img,
                        "point": gt["point"]
                    })

        return tp_all, fp_all, fn_all
