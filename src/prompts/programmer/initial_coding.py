# src/agents/prompts/programmer/initial_coding.py

INITIAL_CODING_PROMPT = """
[INITIAL CODING]

You are implementing the FIRST version of the solution.

RAW INPUT IMAGES:
{raw_images_placeholder}

IMPORTANT:
- Focus on correctness and robustness.
- Implement a clean first version.
- Avoid unnecessary complexity.

{base_prompt}
"""
