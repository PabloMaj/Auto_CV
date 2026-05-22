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

Rules:
- correlate visual problems with code problems
- avoid generic suggestions
- produce concise output
- when providing improvement suggestions include concrete code changes that clearly tell the next agent exactly what to do 
(e.g. increasing a threshold from 10 to 20, reducing kernel size, etc.). The next agent should know precisely what changes to implement.
- Adjust parameters rationally based on observed errors. Avoid large changes that overshoot the optimal value,
as they often lead to missed detections. Prefer small, incremental adjustments rather than aggressive tuning.

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
