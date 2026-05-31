from src.funcs.evaluator_funcs.utils.geometry import compute_iou


class BoxMatcher:

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def match(self, predictions, ground_truths):

        tp_all, fp_all, fn_all = [], [], []

        used_gts = {
            img: [False] * len(gts)
            for img, gts in ground_truths.items()
        }

        for pred in predictions:

            img = pred["image"]
            gts = ground_truths.get(img, [])

            best_iou = 0.0
            best_idx = -1

            for i, gt in enumerate(gts):

                if used_gts[img][i]:
                    continue

                iou = compute_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self.iou_threshold:

                tp_all.append({
                    "image": img,
                    "bbox": pred["bbox"],
                    "gt": gts[best_idx]["bbox"],
                    "score": pred["score"]
                })

                used_gts[img][best_idx] = True

            else:

                fp_all.append({
                    "image": img,
                    "bbox": pred["bbox"],
                    "score": pred["score"]
                })

        # FN
        for img, gts in ground_truths.items():
            for i, gt in enumerate(gts):
                if not used_gts[img][i]:
                    fn_all.append({
                        "image": img,
                        "bbox": gt["bbox"]
                    })

        return tp_all, fp_all, fn_all
