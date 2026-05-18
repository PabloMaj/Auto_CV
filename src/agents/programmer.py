# src/agents/programmer.py

from typing import Dict, List

from src.agents.funcs.programmer import extract_source_code, select_images_for_prompt

from src.agents.funcs.programmer_prompt_builder import ProgrammerPromptBuilder
from src.inference.factory import InferenceFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgrammerAgent:

    CODE_START_TOKEN = "<SOURCE_CODE_START>"
    CODE_END_TOKEN = "<SOURCE_CODE_END>"

    def __init__(self, inference_backend: str, inference_kwargs: Dict, n_to_vis: int = 4):
        self.llm = InferenceFactory.create(backend=inference_backend, **inference_kwargs)
        self.n_to_vis = n_to_vis

    def select_images_for_prompt(self, state) -> List[str]:
        return select_images_for_prompt(state, self.n_to_vis)

    def extract_source_code(self, raw_output: str) -> str:
        return extract_source_code(raw_output, start_token=self.CODE_START_TOKEN, end_token=self.CODE_END_TOKEN,)

    def build_prompt(self, state, reasoning_type) -> str:
        return ProgrammerPromptBuilder.build(state=state, reasoning_type=reasoning_type, code_start_token=self.CODE_START_TOKEN,
                                             code_end_token=self.CODE_END_TOKEN)

    def run(self, state, reasoning_type="initial_coding"):

        print(f"Current stage_id and step_id: {state.get('stage_id', 0)}, {state.get('step_id', 0)}")

        logger.info(f"Running ProgrammerAgent ({reasoning_type})")
        prompt = self.build_prompt(state=state, reasoning_type=reasoning_type)
        messages = self.llm.build_messages(prompt=prompt, image_paths=None)
        raw_output = self.llm.infer(messages=messages)
        generated_code = self.extract_source_code(raw_output)

        state["generated_code_raw"] = raw_output
        state["generated_code"] = generated_code

        return state
