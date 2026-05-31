from src.funcs.evaluator_funcs.evaluators.base import BaseEvaluator
from src.funcs.evaluator_funcs.utils.loaders import load_yolo_points

from src.funcs.evaluator_funcs.matchers.point_matcher import PointMatcher
from src.funcs.evaluator_funcs.metrics.point_metrics import compute_point_metrics
from src.funcs.evaluator_funcs.visualization.point_visualizer import PointVisualizer


class MidpointEvaluator(BaseEvaluator):

    metric_name = "F1"

    # -----------------------------------------
    # GT
    # -----------------------------------------
    def load_gt(self, label_dir, img_stem, img_shape):
        return load_yolo_points(label_dir, img_stem, img_shape)

    # -----------------------------------------
    # MATCHING
    # -----------------------------------------
    def match(self, predictions, ground_truths):

        matcher = PointMatcher(distance_threshold=25)

        return matcher.match(predictions=predictions, ground_truths=ground_truths)

    # -----------------------------------------
    # METRICS
    # -----------------------------------------
    def compute_metric(self, tp_all, fp_all, fn_all, all_gts):

        return compute_point_metrics(tp_all=tp_all, fp_all=fp_all, fn_all=fn_all, all_gts=all_gts)

    # -----------------------------------------
    # VISUALIZATION
    # -----------------------------------------
    def visualize_single(self, image_path, tp_all, fp_all, fn_all):

        return PointVisualizer.visualize(image_path=image_path, tp_all=tp_all, fp_all=fp_all, fn_all=fn_all)
