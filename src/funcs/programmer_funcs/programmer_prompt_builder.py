# src/agents/funcs/programmer_prompt_builder.py

from src.prompts.programmer_prompts.base import BASE_PROGRAMMER_PROMPT
from src.prompts.programmer_prompts.initial_coding import INITIAL_CODING_PROMPT
from src.prompts.programmer_prompts.bug_fixing import BUG_FIXING_PROMPT
from src.prompts.programmer_prompts.improving_based_on_suggestion import IMPROVING_BASED_ON_SUGGESTION_PROMPT
from src.prompts.programmer_prompts.novelty_coding import NOVELTY_CODING_PROMPT

from enum import Enum


class ProgrammerReasoningType(str, Enum):
    INITIAL_CODING = "initial_coding"
    BUG_FIXING = "bug_fixing"
    IMPROVING_BASED_ON_SUGGESTION = "improving_based_on_suggestion"
    NOVELTY_CODING = "novelty_coding"


class ProgrammerPromptBuilder:

    @classmethod
    def build(cls, state, reasoning_type, code_start_token, code_end_token):

        base_prompt = BASE_PROGRAMMER_PROMPT.format(
            user_prompt=state["user_prompt"],
            desired_output_specification=state["desired_output_definition"],
            code_start_token=code_start_token,
            code_end_token=code_end_token,
            model_path=state.get("yolo_model_path", "")
        )

        if reasoning_type == ProgrammerReasoningType.INITIAL_CODING:
            return INITIAL_CODING_PROMPT.format(
                raw_images_placeholder=state.get("raw_images_section", ""),
                base_prompt=base_prompt,
            )

        elif reasoning_type == ProgrammerReasoningType.BUG_FIXING:
            return BUG_FIXING_PROMPT.format(
                previous_code=state["generated_code"],
                runner_error=state["runner_error"],
                base_prompt=base_prompt,
            )

        elif reasoning_type == ProgrammerReasoningType.IMPROVING_BASED_ON_SUGGESTION:
            return IMPROVING_BASED_ON_SUGGESTION_PROMPT.format(
                previous_code=state["generated_code"],
                verified_problems=state["verified_problems"],
                improvement_suggestions=state["improvement_suggestions"],
                base_prompt=base_prompt,
            )

        elif reasoning_type == ProgrammerReasoningType.NOVELTY_CODING:
            return NOVELTY_CODING_PROMPT.format(
                previous_implementations=state.get("previous_implementations", ""),
                base_prompt=base_prompt,
            )

        raise ValueError(f"Unknown reasoning type: {reasoning_type}")
