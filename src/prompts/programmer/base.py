# src/agents/prompts/programmer/base.py

BASE_PROGRAMMER_PROMPT = """
USER TASK:
{user_prompt}

DESIRED OUTPUT SPECIFICATION:
{desired_output_specification}

GENERAL REQUIREMENTS:
- Return ONLY Python source code.
- Wrap the source code between:
  {code_start_token}
  and
  {code_end_token}

IMPLEMENTATION REQUIREMENTS:
- Main implementation must be inside class Predictor.
- Additional helper classes and functions are allowed.
- Saveable as generated_solution.py.
- Must contain:

if __name__ == "__main__":

- Code must be executable.
- No pseudocode.
- No TODO placeholders.
- Include imports.
- Prefer robust and readable implementations.
- Handle edge cases.
- Avoid unnecessary complexity.
"""
