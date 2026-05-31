from src.utils.logger import get_logger
from src.funcs.evaluator_funcs.evaluators.factory import EvaluatorFactory
from src.funcs.evaluator_funcs.evaluators.placeholder import placeholder_evaluation

logger = get_logger(__name__)


class EvaluatorAgent:

    def run(self, state):
        logger.info("Running EvaluatorAgent")

        desired_output = state.get("desired_output", "unknown")

        evaluator = EvaluatorFactory.create(desired_output)

        if evaluator is None:
            state = placeholder_evaluation(state, desired_output)
            logger.warning(f"No evaluator for type: {desired_output}")
            return state

        state = evaluator.evaluate(state)

        logger.info("EvaluatorAgent finished successfully")

        print(f"Evaluation results: {state.get('evaluation', {})}")

        return state
