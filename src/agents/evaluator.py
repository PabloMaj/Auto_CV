
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluatorAgent:

    def run(self, state):
        logger.info("Running EvaluatorAgent")

        # TODO:
        # - inference on validation set
        # - calculate metrics
        # - visualize TP/FP/FN

        state["evaluation_metric"] = 0.75
        state["evaluation_summary"] = "Prototype metric"

        state["prediction_visualizations"] = [
            "outputs/tp_example.png",
            "outputs/fp_example.png",
            "outputs/fn_example.png"
        ]

        return state
