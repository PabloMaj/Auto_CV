LLM_JUDGE_SYSTEM_PROMPT = """
You are a Computer Vision evaluation expert.
Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block.
"""

LLM_JUDGE_USER_TEMPLATE = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the {n_images} attached image(s).

F1 score measures detection quality by balancing two aspects:
  — recall:    how many real {output_type} are covered by a CYAN prediction
  — precision: how many CYAN predictions land on a real {output_type}
F1 is low if either aspect is poor; F1 is high only when both are good.

Evaluate across all provided images together and return a single score.

Estimate:
  recall:       fraction of real {output_type} with a CYAN prediction (0–1)
  precision:    fraction of CYAN predictions on a real {output_type} (0–1)
  metric_value: the F1 score (0 = worst, 1 = perfect)

Return ONLY:
{{
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float>
}}
"""
