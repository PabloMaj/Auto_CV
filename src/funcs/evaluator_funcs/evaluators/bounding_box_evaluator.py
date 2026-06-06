from src.funcs.evaluator_funcs.evaluators.base import BaseEvaluator
from src.funcs.evaluator_funcs.utils.loaders import load_yolo_boxes
from src.funcs.evaluator_funcs.matchers.box_matcher import BoxMatcher
from src.funcs.evaluator_funcs.metrics.ap50 import compute_ap50
from src.funcs.evaluator_funcs.visualization.box_visualizer import BoxVisualizer


class BoundingBoxEvaluator(BaseEvaluator):

    metric_name = "BOX_F1"

    def load_gt(self, label_dir, img_stem, img_shape):
        return load_yolo_boxes(label_dir, img_stem, img_shape)

    def match(self, predictions, ground_truths):
        return BoxMatcher(iou_threshold=0.5).match(
            predictions=predictions,
            ground_truths=ground_truths,
        )

    def compute_metric(self, tp_all, fp_all, fn_all, all_gts):
        total_gt = sum(len(v) for v in all_gts.values())
        tp = len(tp_all)
        fp = len(fp_all)
        fn = len(fn_all)

        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        ap50 = compute_ap50(tp_all=tp_all, fp_all=fp_all, total_gt=total_gt)

        return {
            "metric_name": "BOX_F1",
            "metric_value": round(f1, 4),   # optimization signal
            "ap50": round(ap50, 4),
            "status": "success",
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def visualize_single(self, image_path, tp_all, fp_all, fn_all):
        return BoxVisualizer.visualize(
            image_path=image_path,
            tp_all=tp_all,
            fp_all=fp_all,
            fn_all=fn_all,
        )
