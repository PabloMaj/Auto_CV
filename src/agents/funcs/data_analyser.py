"""
Helper functions for DataAnalyserAgent
"""

from src.utils.logger import get_logger
from src.prompts.data_analyser import DESIRED_OUTPUT_SYSTEM_PROMPT, DESIRED_OUTPUT_USER_PROMPT

logger = get_logger(__name__)


def determine_desired_output(inference, user_prompt: str, allowed_outputs: list):
    """
    Determine the most appropriate prediction output format for a CV task.

    Args:
        inference: Inference engine instance
        user_prompt: User's task description
        allowed_outputs: List of allowed output formats

    Returns:
        str: Selected output format
    """

    prompt = DESIRED_OUTPUT_USER_PROMPT.format(user_prompt=user_prompt)
    messages = inference.build_messages(prompt=prompt, system_prompt=DESIRED_OUTPUT_SYSTEM_PROMPT)
    response = inference.infer(messages=messages, options={"temperature": 0.0})
    response = response.strip()

    if response not in allowed_outputs:
        logger.warning(f"Invalid desired_output response: {response}")
        return "unknown"

    return response
