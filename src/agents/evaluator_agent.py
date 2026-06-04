from src.utils.logger import get_logger
from src.funcs.evaluator_funcs.evaluators.factory import EvaluatorFactory
from src.funcs.evaluator_funcs.evaluators.placeholder import placeholder_evaluation
from src.funcs.evaluator_funcs.evaluators.llm_judge_val_evaluator import LLMJudgeValEvaluator
from src.inference.factory import InferenceFactory

logger = get_logger(__name__)


class EvaluatorAgent:

    def __init__(self, settings=None):
        self.settings = settings
        self.llm_judge = None

        if settings and settings.enable_label_free_improvement:
            llm = InferenceFactory.create(
                backend=settings.improvement_llm.backend,
                model=settings.improvement_llm.model,
                **settings.improvement_llm.inference_kwargs,
            )
            self.llm_judge = LLMJudgeValEvaluator(llm)
            logger.info("EvaluatorAgent: label-free mode enabled (LLM judge for val)")

    def run(self, state):
        logger.info("Running EvaluatorAgent")

        desired_output = state.get("desired_output", "unknown")
        evaluator = EvaluatorFactory.create(desired_output)

        if evaluator is None:
            state = placeholder_evaluation(state, desired_output)
            logger.warning(f"No evaluator for type: {desired_output}")
            return state

        if self.llm_judge:
            # label-free: standard eval on test, LLM judge on val
            state = evaluator.evaluate_test_split_only(state)
            state = self.llm_judge.evaluate_val(state)
        else:
            state = evaluator.evaluate(state)

        logger.info("EvaluatorAgent finished successfully")
        print(f"Evaluation results: {state.get('evaluation', {})}")
        return state
