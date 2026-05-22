IMPROVING_BASED_ON_SUGGESTION_PROMPT = """
[IMPROVING BASED ON SUGGESTIONS]

You are improving an existing implementation based on verified feedback.

PREVIOUS IMPLEMENTATION:
{previous_code}

VERIFIED PROBLEMS:
{verified_problems}

SUGGESTED IMPROVEMENTS:
{improvement_suggestions}

IMPORTANT:
- Apply improvements carefully.
- Do not break already working functionality.
- Improve quality, robustness, and performance where possible.

{base_prompt}
"""
