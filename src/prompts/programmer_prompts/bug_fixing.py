BUG_FIXING_PROMPT = """
[BUG FIXING]

You are fixing a previously generated implementation.

PREVIOUS IMPLEMENTATION:
{previous_code}

RUNNER ERROR:
{runner_error}

IMPORTANT:
- Focus ONLY on fixing the issue.
- Preserve working parts of the implementation.
- Ensure the code runs successfully.

{base_prompt}
"""
