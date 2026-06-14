"""
Prompt variants v3 — informed by rounds 1 & 2.

Round summary (bboxes, 20 cases each):
  v1  pr_decomposed   R²=+0.50   P/R decomposition + formula
  v2  ordinal_f1      R²=+0.58   coarse bucket → refine  ← overall best so far
  v2  proportional_pr R²=+0.41   coverage% + noise% estimation
  v2  pr_guided       R²=-0.00   extra visual cues HURT vs bare pr_decomposed
  v1  split_errors    R²=-0.67   separate missed/wrong fractions fail
  v1  baseline        R²=-1.15   strict anchored scale inverts correlation
  v1  cot_counting    R²=-7.83   explicit object counting is catastrophic

Design principles for v3
------------------------
1. Ordinal (coarse→fine) is the strongest pattern — enhance it further.
2. Proportion % estimation (no counting) is consistently positive.
3. Keep prompts SHORT — verbose guidance degrades calibration (pr_guided lesson).
4. Avoid strict/negative anchoring — it creates inverse correlation.
5. Avoid showing formula derivation steps — arithmetic errors hurt scores.
6. Test: separate ordinal for P and R independently, then compute F1.
7. Test: just ask 2 % questions and let model output metric_value directly.
"""

_JSON_ONLY = (
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block."
)

# ---------------------------------------------------------------------------
# V3-1: ORDINAL_VISUAL
# Extends ordinal_f1 (v2 winner) with UAV-specific visual descriptions per level.
# Hypothesis: concrete visual anchors improve level selection accuracy.
# ---------------------------------------------------------------------------
ORDINAL_VISUAL_SYSTEM = f"""
You are a Computer Vision expert evaluating UAV crop detection.
You classify detection quality into a level and then refine to a score.

{_JSON_ONLY}
"""

ORDINAL_VISUAL_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions are drawn on the attached UAV crop image.

Pick the level that best matches what you see:

LEVEL A (F1 ≈ 0.90–1.0)
  Nearly every visible plant/row has a CYAN mark. Bare soil and gaps between
  rows are free of CYAN. At most 1–2 spurious predictions visible.

LEVEL B (F1 ≈ 0.70–0.90)
  Most plants/rows covered by CYAN, but a few are missed OR a few CYAN marks
  sit on empty ground. Not both problems together.

LEVEL C (F1 ≈ 0.50–0.70)
  Roughly half covered. Either a noticeable fraction of plants are missed, OR
  a noticeable fraction of predictions land on empty ground (or both mildly).

LEVEL D (F1 ≈ 0.30–0.50)
  Majority of plants missed OR majority of predictions are spurious.

LEVEL E (F1 ≈ 0.0–0.30)
  Almost nothing detected, or nearly all predictions are wrong.

Then set metric_value to your precise F1 estimate within that level's range.

Return ONLY this JSON:
{{
  "level": "<A/B/C/D/E>",
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# V3-2: PR_MINIMAL
# Bare-minimum P/R prompt — strip everything that hurt pr_guided.
# Tests whether simplest possible P/R instruction reproduces v1 pr_decomposed (R²=0.50).
# ---------------------------------------------------------------------------
PR_MINIMAL_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_MINIMAL_USER = """
TASK: {user_prompt}  OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV image.

Estimate:
  recall    = fraction of real {output_type} that have a CYAN prediction on them  (0–1)
  precision = fraction of CYAN predictions that land on a real {output_type}        (0–1)
  metric_value = 2 × recall × precision / (recall + precision)

Return ONLY:
{{
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float>
}}
"""

# ---------------------------------------------------------------------------
# V3-3: DUAL_ORDINAL
# Apply the ordinal approach separately to recall AND precision, then compute F1.
# Hypothesis: separating P from R reduces interference; ordinal beats continuous.
# ---------------------------------------------------------------------------
DUAL_ORDINAL_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You classify recall and precision independently before computing F1.

{_JSON_ONLY}
"""

DUAL_ORDINAL_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV crop image.

Classify RECALL (coverage of real objects):
  R-HIGH   ≈ 0.90   almost all real {output_type} have a CYAN mark
  R-GOOD   ≈ 0.75   most covered, a few missed
  R-MED    ≈ 0.55   roughly half covered
  R-LOW    ≈ 0.35   majority missed
  R-POOR   ≈ 0.15   very few detected

Classify PRECISION (correctness of predictions):
  P-HIGH   ≈ 0.90   almost all CYAN marks land on real {output_type}
  P-GOOD   ≈ 0.75   most correct, a few spurious
  P-MED    ≈ 0.55   roughly half correct
  P-LOW    ≈ 0.35   majority spurious
  P-POOR   ≈ 0.15   almost all spurious

Set metric_value = 2 × recall × precision / (recall + precision).

Return ONLY this JSON:
{{
  "recall_class": "<R-HIGH/R-GOOD/R-MED/R-LOW/R-POOR>",
  "precision_class": "<P-HIGH/P-GOOD/P-MED/P-LOW/P-POOR>",
  "recall": <float — midpoint of chosen class>,
  "precision": <float — midpoint of chosen class>,
  "metric_value": <float F1>
}}
"""

# ---------------------------------------------------------------------------
# V3-4: PROPORTION_DIRECT
# Ask 2 proportion questions (like proportional_pr v2) but skip showing the
# formula — model outputs metric_value directly instead of computing it.
# Hypothesis: removing arithmetic derivation reduces compounding errors.
# ---------------------------------------------------------------------------
PROPORTION_DIRECT_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PROPORTION_DIRECT_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV crop image.

Answer two questions as percentages (0–100):

Q1 COVERAGE: Of all real {output_type} visible in the scene, what percentage
have a CYAN prediction on or very near them?

Q2 NOISE: Of all CYAN predictions, what percentage land on empty background
with no real {output_type} nearby?

Then set metric_value to the F1 score that corresponds to those two values
(high coverage + low noise → high F1; either problem → lower F1).

Return ONLY this JSON:
{{
  "coverage_pct": <int 0–100>,
  "noise_pct": <int 0–100>,
  "metric_value": <float 0.0–1.0>
}}
"""

# ---------------------------------------------------------------------------
# V3-5: HOLISTIC_CALIBRATED
# Single direct F1 question with positive (not strict) calibration anchors.
# Tests whether well-framed holistic estimation beats decomposition.
# Avoids the strict/negative anchoring that caused baseline's -1.15 R².
# ---------------------------------------------------------------------------
HOLISTIC_CALIBRATED_SYSTEM = f"""
You are a Computer Vision evaluation expert for UAV agricultural imagery.
{_JSON_ONLY}
"""

HOLISTIC_CALIBRATED_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV crop image.

Estimate the F1 score (0.0–1.0) that best reflects detection quality.
F1 balances how many real objects are found with how many predictions are correct.

Calibration anchors (positive framing):
  1.0 → all real objects found, all predictions are on real objects
  0.8 → most objects found AND most predictions correct
  0.6 → about 60–70% found AND about 60–70% predictions correct
  0.4 → significant fraction missed OR significant fraction spurious
  0.0 → nothing useful detected

Return ONLY this JSON:
{{
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# Public dict consumed by run.py
# ---------------------------------------------------------------------------
PROMPTS_V3 = {
    "ordinal_visual": {
        "label": "Ordinal + Visual Anchors",
        "system": ORDINAL_VISUAL_SYSTEM,
        "user_template": ORDINAL_VISUAL_USER,
    },
    "pr_minimal": {
        "label": "P/R Minimal (3 lines)",
        "system": PR_MINIMAL_SYSTEM,
        "user_template": PR_MINIMAL_USER,
    },
    "dual_ordinal": {
        "label": "Dual Ordinal (P + R separately)",
        "system": DUAL_ORDINAL_SYSTEM,
        "user_template": DUAL_ORDINAL_USER,
    },
    "proportion_direct": {
        "label": "Proportion % → Direct F1",
        "system": PROPORTION_DIRECT_SYSTEM,
        "user_template": PROPORTION_DIRECT_USER,
    },
    "holistic_calibrated": {
        "label": "Holistic Calibrated (positive anchors)",
        "system": HOLISTIC_CALIBRATED_SYSTEM,
        "user_template": HOLISTIC_CALIBRATED_USER,
    },
}
