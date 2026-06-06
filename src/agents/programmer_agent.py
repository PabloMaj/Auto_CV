from pathlib import Path
from typing import Dict, List

from src.funcs.programmer_funcs.programmer_funcs import extract_source_code, select_images_for_prompt
from src.funcs.programmer_funcs.programmer_prompt_builder import ProgrammerPromptBuilder
from src.funcs.programmer_funcs.training_image_sampler import sample_training_images
from src.inference.factory import InferenceFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgrammerAgent:

    CODE_START_TOKEN = "<SOURCE_CODE_START>"
    CODE_END_TOKEN = "<SOURCE_CODE_END>"

    def __init__(self, inference_backend: str, inference_kwargs: Dict,
                 n_to_vis: int = 4, label_free: bool = False):
        self.llm = InferenceFactory.create(backend=inference_backend, **inference_kwargs)
        self.n_to_vis = n_to_vis
        self.label_free = label_free

    def select_images_for_prompt(self, state) -> List[str]:
        return select_images_for_prompt(state, self.n_to_vis)

    def extract_source_code(self, raw_output: str) -> str:
        return extract_source_code(
            raw_output,
            start_token=self.CODE_START_TOKEN,
            end_token=self.CODE_END_TOKEN,
        )

    def build_prompt(self, state, reasoning_type) -> str:
        return ProgrammerPromptBuilder.build(
            state=state,
            reasoning_type=reasoning_type,
            code_start_token=self.CODE_START_TOKEN,
            code_end_token=self.CODE_END_TOKEN,
            label_free=self.label_free,
        )

    def run(self, state, reasoning_type="initial_coding"):
        stage_id = state.get("stage_id", 0)
        step_id = state.get("step_id", 0)
        print(f"Current stage_id and step_id: {stage_id}, {step_id}")

        repo_root = Path(__file__).resolve().parents[2]
        self.llm.log_dir = repo_root / "workspace" / state.get("exp_id", "default") / "llm_logs" / "programmer" / f"stage_{stage_id}_step_{step_id}"

        logger.info(f"Running ProgrammerAgent ({reasoning_type})")

        prompt = self.build_prompt(state=state, reasoning_type=reasoning_type)

        image_paths = None
        if reasoning_type == "initial_coding":
            image_paths = sample_training_images(state, label_free=self.label_free) or None

        messages = self.llm.build_messages(prompt=prompt, image_paths=image_paths)
        raw_output = self.llm.infer(messages=messages)
        generated_code = self.extract_source_code(raw_output)

        state["generated_code_raw"] = raw_output
        state["generated_code"] = generated_code

        return state
