from src.funcs.evaluator_funcs.evaluators.base import BaseEvaluator
from src.funcs.evaluator_funcs.utils.loaders import load_yolo_lines

from src.funcs.evaluator_funcs.matchers.line_matcher import LineMatcher
from src.funcs.evaluator_funcs.metrics.line_metrics import compute_line_metrics
from src.funcs.evaluator_funcs.visualization.line_visualizer import LineVisualizer


class LineEvaluator(BaseEvaluator):

    metric_name = "LINE_F1"

    # -----------------------------------------
    # GT
    # -----------------------------------------
    def load_gt(self, label_dir, img_stem, img_shape):
        return load_yolo_lines(label_dir, img_stem, img_shape)

    # -----------------------------------------
    # MATCHING
    # -----------------------------------------
    def match(self, predictions, ground_truths):

        matcher = LineMatcher(distance_threshold=0.05, angle_threshold=10.0)

        return matcher.match(predictions=predictions, ground_truths=ground_truths)

    # -----------------------------------------
    # METRICS
    # -----------------------------------------
    def compute_metric(self, tp_all, fp_all, fn_all, all_gts):

        return compute_line_metrics(tp_all=tp_all, fp_all=fp_all, fn_all=fn_all, all_gts=all_gts)

    # -----------------------------------------
    # VISUALIZATION
    # -----------------------------------------
    def visualize_single(self, image_path, tp_all, fp_all, fn_all):

        return LineVisualizer.visualize(image_path=image_path, tp_all=tp_all, fp_all=fp_all, fn_all=fn_all)
