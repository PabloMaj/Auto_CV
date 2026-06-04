from src.prompts.improvement_suggester_prompts.improvement_suggester_prompt import (
    VERIFIED_PROBLEMS_START,
    VERIFIED_PROBLEMS_END,
    IMPROVEMENT_SUGGESTIONS_START,
    IMPROVEMENT_SUGGESTIONS_END,
)

LABEL_FREE_SYSTEM_PROMPT = f"""
You are an elite Computer Vision Engineer and Python Developer.

Your task is to analyze:

1. Computer vision task description
2. Current source code
3. Prediction visualization images assessed by an LLM judge

The attached images show model predictions on validation images.
All predictions are drawn in CYAN — no ground-truth labels are available.

Focus on patterns visible in the predictions:
e.g. lines that don't follow crop rows, missing detections, mis-aligned bounding boxes.

Rules:
- correlate visual patterns with code problems
- avoid generic suggestions
- produce concise output
- when providing improvement suggestions include concrete code changes that clearly tell
  the next agent exactly what to do (e.g. increasing a threshold from 10 to 20,
  reducing kernel size, etc.)
- prefer small, incremental adjustments over large parameter sweeps

Return output ONLY in the following format.

{VERIFIED_PROBLEMS_START}
- problem 1
- problem 2
{VERIFIED_PROBLEMS_END}

{IMPROVEMENT_SUGGESTIONS_START}
- suggestion 1
- suggestion 2
{IMPROVEMENT_SUGGESTIONS_END}
"""


def build_label_free_user_prompt(user_prompt: str, generated_code: str) -> str:
    return f"""
# COMPUTER VISION TASK

{user_prompt}

# CURRENT SOURCE CODE

```python
{generated_code}
```

Analyze BOTH:
 - source code
 - attached prediction visualization images (colours estimated by LLM judge, not GT labels)

The attached images show predictions in CYAN (no ground-truth labels available).
""".strip()
