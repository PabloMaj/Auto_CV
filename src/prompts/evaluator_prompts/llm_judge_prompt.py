LLM_JUDGE_SYSTEM_PROMPT = """
You are a Computer Vision evaluation expert.
Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block.
"""

LLM_JUDGE_USER_TEMPLATE = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the {n_images} attached image(s).

Step 1 — Count individual instances.
Before scoring, identify each individual {output_type} instance visible in the image.
Each real object counts as a separate instance — a single CYAN prediction that spans
multiple objects is ONE prediction, not multiple. This matters: one oversized prediction
covering 3 objects produces 1 FP and 3 FN, which severely penalises the score.
Similarly, for lines or midpoints: if several CYAN predictions are clustered very close
together (nearly overlapping), they likely represent the same underlying instance — all
but one would count as false positives.

If no CYAN predictions are visible in any of the images, return metric_value = 0.0 immediately.

Step 2 — Estimate F1.
F1 score measures detection quality by balancing two aspects:
  — recall:    how many real {output_type} are covered by a CYAN prediction (0–1)
  — precision: how many CYAN predictions land on a real {output_type} (0–1)
F1 is low if either aspect is poor; F1 is high only when both are good.

Evaluate across all provided images together and return a single score.

Return ONLY:
{{
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float>
}}
"""
