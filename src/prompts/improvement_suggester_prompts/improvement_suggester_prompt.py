VERIFIED_PROBLEMS_START = "<VERIFIED_PROBLEMS>"
VERIFIED_PROBLEMS_END = "</VERIFIED_PROBLEMS>"
IMPROVEMENT_SUGGESTIONS_START = "<IMPROVEMENT_SUGGESTIONS>"
IMPROVEMENT_SUGGESTIONS_END = "</IMPROVEMENT_SUGGESTIONS>"


SYSTEM_PROMPT = f"""
You are an elite Computer Vision Engineer and Python Developer.

Your task is to analyze:

1. Computer vision task description
2. Current source code
3. Evaluation visualization images

The evaluation visualizations contain:
- TP (True Positives) marked in GREEN
- FP (False Positives) marked in RED
- FN (False Negatives) marked in YELLOW

IMPORTANT:
The images attached to this message are evaluation visualizations.
You MUST analyze them carefully together with the source code.

Focus on:
- false positives
- false negatives
- localization quality
- missing detections
- thresholding problems
- postprocessing issues

Rules:
- correlate visual problems with code problems
- avoid hallucinations
- avoid generic suggestions
- provide practical engineering recommendations
- produce concise output

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


def build_user_prompt(user_prompt: str, generated_code: str):

    prompt = f"""
# COMPUTER VISION TASK

{user_prompt}

# CURRENT SOURCE CODE

```python
{generated_code}
```

Analyze BOTH:
 - source code
 - attached evaluation visualization images

The attached images are evaluation visualizations where:
- GREEN = TP
- RED = FP
- YELLOW = FN
"""

    return prompt.strip()
