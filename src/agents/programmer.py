# src/agents/programmer.py

import random
import re
from typing import Dict, List

from src.inference.factory import InferenceFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProgrammerAgent:

    CODE_START_TOKEN = "<SOURCE_CODE_START>"
    CODE_END_TOKEN = "<SOURCE_CODE_END>"

    def __init__(
        self,
        inference_backend: str,
        inference_kwargs: Dict,
        n_to_vis: int = 4
    ):

        self.llm = InferenceFactory.create(
            backend=inference_backend,
            **inference_kwargs
        )

        self.n_to_vis = n_to_vis

    # ======================================================
    # IMAGE SELECTION
    # ======================================================

    def select_images_for_prompt(
        self,
        state
    ) -> List[str]:

        train_images = state.get(
            "path_to_train_images",
            []
        )

        if not train_images:
            return []

        if len(train_images) <= self.n_to_vis:
            return train_images

        return random.sample(
            train_images,
            self.n_to_vis
        )

    # ======================================================
    # CODE EXTRACTION
    # ======================================================

    def extract_source_code(
        self,
        raw_output: str
    ) -> str:

        if not raw_output:
            return ""

        code = raw_output.strip()

        start_token_name = re.sub(r"^<|>$", "", self.CODE_START_TOKEN)
        end_token_name = re.sub(r"^<|>$", "", self.CODE_END_TOKEN)
        tagged_pattern = (
            rf"{re.escape(self.CODE_START_TOKEN)}"
            rf"(.*?)"
            rf"(?:{re.escape(self.CODE_END_TOKEN)}|</{re.escape(end_token_name)}>|</{re.escape(start_token_name)}>)"
        )

        tagged_match = re.search(
            tagged_pattern,
            code,
            re.DOTALL
        )

        if tagged_match:
            return tagged_match.group(1).strip()

        markdown_python_pattern = (
            r"```python\s*(.*?)```"
        )

        markdown_match = re.search(
            markdown_python_pattern,
            code,
            re.DOTALL | re.IGNORECASE
        )

        if markdown_match:
            return markdown_match.group(1).strip()

        generic_markdown_pattern = (
            r"```\s*(.*?)```"
        )

        generic_match = re.search(
            generic_markdown_pattern,
            code,
            re.DOTALL
        )

        if generic_match:
            return generic_match.group(1).strip()

        return code

    # ======================================================
    # EXECUTION FEEDBACK
    # ======================================================

    def build_execution_feedback_section(
        self,
        state
    ) -> str:

        execution_feedback = state.get(
            "execution_feedback",
            {}
        )

        if not execution_feedback:
            return "No execution feedback available."

        success = execution_feedback.get(
            "success",
            None
        )

        return_code = execution_feedback.get(
            "return_code",
            None
        )

        stdout = execution_feedback.get(
            "stdout",
            ""
        )

        stderr = execution_feedback.get(
            "stderr",
            ""
        )

        return f"""
Success:
{success}

Return code:
{return_code}

STDOUT:
{stdout}

STDERR:
{stderr}
"""

    # ======================================================
    # PROMPT
    # ======================================================

    def build_prompt(
        self,
        state,
        reasoning_type
    ) -> str:

        user_prompt = state.get(
            "user_prompt",
            ""
        )

        vision_analysis = state.get(
            "vision_analysis",
            ""
        )

        improvement_suggestions = state.get(
            "improvement_suggestions",
            ""
        )

        execution_feedback = (
            self.build_execution_feedback_section(
                state
            )
        )

        sample_image_path = ""
        image_paths = state.get("path_to_train_images", [])
        if image_paths:
            sample_image_path = random.choice(image_paths)
            state["sample_image_path"] = sample_image_path

        sample_image_section = ""
        if sample_image_path:
            sample_image_section = f"""
==========================================================
SAMPLE IMAGE PATH
==========================================================

Use this sample image path from the training dataset for inference/testing:
{sample_image_path}
"""

        prompt = f"""
You are an elite Computer Vision Engineer.

Generate executable production-ready Python code.

==========================================================
USER REQUEST
==========================================================

{user_prompt}

==========================================================
VISION ANALYSIS
==========================================================

{vision_analysis}

==========================================================
PREVIOUS IMPROVEMENT SUGGESTIONS
==========================================================

{improvement_suggestions}

==========================================================
EXECUTION FEEDBACK
==========================================================

{execution_feedback}

{sample_image_section}

==========================================================
IMPLEMENTATION REQUIREMENTS
==========================================================

Requirements:

1. Implement a complete solution

2. Main class MUST be:

Predictor

3. Include:
- preprocessing
- inference
- postprocessing

4. Use robust engineering practices

5. Prefer:
- OpenCV
- NumPy
- scikit-image

6. Avoid:
- placeholders
- TODO
- pseudo-code
- incomplete methods

7. The code MUST be executable directly

8. Minimize unnecessary dependencies

9. Save any inference visualization to the workspace as `output_visualization.jpg`

==========================================================
BUGFIX INSTRUCTIONS
==========================================================

If execution previously failed:

- identify root cause
- minimally modify broken logic
- preserve working components
- avoid rewriting entire pipeline
- fix runtime and syntax issues carefully

==========================================================
OUTPUT FORMAT
==========================================================

Return source code ONLY between:

{self.CODE_START_TOKEN}

and

{self.CODE_END_TOKEN}

Do NOT use markdown.
Do NOT explain anything.
"""

        if reasoning_type == "initial_coding":

            prompt += """

CURRENT OBJECTIVE:
Generate initial working solution.
"""

        elif reasoning_type == "bugfix":

            prompt += """

CURRENT OBJECTIVE:
Fix execution/runtime/syntax errors.
Focus on minimal targeted fixes.
"""

        elif reasoning_type == "improvement":

            prompt += """

CURRENT OBJECTIVE:
Improve robustness and counting accuracy.
Preserve existing working pipeline.
"""

        return prompt

    # ======================================================
    # RUN
    # ======================================================

    def run(
        self,
        state,
        reasoning_type="initial_coding"
    ):

        logger.info(
            f"Running ProgrammerAgent "
            f"({reasoning_type})"
        )

        prompt = self.build_prompt(
            state=state,
            reasoning_type=reasoning_type
        )

        messages = self.llm.build_messages(
            prompt=prompt,
            image_paths=None
        )

        raw_output = self.llm.infer(
            messages=messages
        )

        generated_code = (
            self.extract_source_code(
                raw_output
            )
        )

        state["generated_code_raw"] = (
            raw_output
        )

        state["generated_code"] = (
            generated_code
        )

        return state
