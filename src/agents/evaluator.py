# src/agents/evaluator.py

from src.utils.logger import get_logger
from src.agents.funcs.evaluator import evaluate_bounding_boxes, placeholder_evaluation

logger = get_logger(__name__)


class EvaluatorAgent:

    def run(self, state):
        logger.info("Running EvaluatorAgent")

        desired_output = state.get("desired_output", "unknown")

        if desired_output == "bounding_boxes":
            state = evaluate_bounding_boxes(state)
        else:
            state = placeholder_evaluation(state, desired_output)

        logger.info("EvaluatorAgent finished successfully")
        import sys
        sys.exit(0)

        return state
