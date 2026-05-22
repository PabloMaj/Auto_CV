from src.utils.logger import get_logger

from src.inference.factory import InferenceFactory
from src.prompts.improvement_suggester_prompts.improvement_suggester_prompt import SYSTEM_PROMPT, build_user_prompt
from src.funcs.improvement_suggester_funcs.improvement_suggester_functions import parse_improvement_response

logger = get_logger(__name__)


class ImprovementSuggesterAgent:

    def __init__(self, inference_backend: str, inference_kwargs: dict):
        self.llm = InferenceFactory.create(backend=inference_backend, **inference_kwargs)

    def run(self, state):

        logger.info("Running ImprovementSuggesterAgent")

        user_prompt = state.get("user_prompt", "")
        generated_code = state.get("generated_code", "")

        # Choose viusualization from best performing step
        stage_id = state.get("stage_id", 0)
        step_id = state.get("step_id", 0)
        filtered = [a for a in state["eval_artifacts"] if a.step_key.startswith(f"stage_{stage_id}_")]
        best = max(filtered, key=lambda a: a.value, default=None)
        vis_paths = best.img_paths[:2] if best else []
        logger.info(f"For current stage and step {stage_id}_{step_id} we choose mistakes visualizations from {best.step_key if best else 'N/A'} with metric {best.value if best else 'N/A'} -> {vis_paths}")

        prompt = build_user_prompt(user_prompt=user_prompt, generated_code=generated_code)
        messages = self.llm.build_messages(prompt=prompt, image_paths=vis_paths, system_prompt=SYSTEM_PROMPT)
        response = self.llm.infer(messages=messages)

        logger.info("Improvement analysis completed")

        parsed = parse_improvement_response(response)

        state["verified_problems"] = parsed["verified_problems"]
        state["improvement_suggestions"] = parsed["improvement_suggestions"]
        state["improvement_raw_response"] = response

        return state
