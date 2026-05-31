# evaluators/factory.py

from src.funcs.evaluator_funcs.evaluators.bounding_box_evaluator import BoundingBoxEvaluator
from src.funcs.evaluator_funcs.evaluators.midpoint_evaluator import MidpointEvaluator
from src.funcs.evaluator_funcs.evaluators.line_evaluator import LineEvaluator


class EvaluatorFactory:

    _registry = {
        "bounding_boxes": BoundingBoxEvaluator,
        "points": MidpointEvaluator,
        "line_segments": LineEvaluator,
    }

    @staticmethod
    def create(output_type: str):
        evaluator_cls = EvaluatorFactory._registry.get(output_type)

        if evaluator_cls is None:
            return None

        return evaluator_cls()
