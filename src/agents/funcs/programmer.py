"""
Helper functions for ProgrammerAgent
"""

import random
import re
from typing import List


def select_images_for_prompt(state, n_to_vis=4) -> List[str]:
    train_images = state.get("path_to_train_images", [])

    if not train_images:
        return []

    if len(train_images) <= n_to_vis:
        return train_images

    return random.sample(train_images, n_to_vis)


def extract_source_code(raw_output: str, start_token: str, end_token: str) -> str:
    if not raw_output:
        return ""

    code = raw_output.strip()
    start_token_name = re.sub(r"^<|>$", "", start_token)
    end_token_name = re.sub(r"^<|>$", "", end_token)
    tagged_pattern = (
        rf"{re.escape(start_token)}"
        rf"(.*?)"
        rf"(?:{re.escape(end_token)}|</{re.escape(end_token_name)}>|</{re.escape(start_token_name)}>)"
    )

    tagged_match = re.search(tagged_pattern, code, re.DOTALL)
    if tagged_match:
        return tagged_match.group(1).strip()

    markdown_python_pattern = r"```python\s*(.*?)```"
    markdown_match = re.search(markdown_python_pattern, code, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        return markdown_match.group(1).strip()

    generic_markdown_pattern = r"```\s*(.*?)```"
    generic_match = re.search(generic_markdown_pattern, code, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    return code


def build_execution_feedback_section(state) -> str:
    execution_feedback = state.get("execution_feedback", {})
    if not execution_feedback:
        return "No execution feedback available."

    success = execution_feedback.get("success", None)
    return_code = execution_feedback.get("return_code", None)
    stdout = execution_feedback.get("stdout", "")
    stderr = execution_feedback.get("stderr", "")

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
