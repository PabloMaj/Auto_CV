"""
Prompt variants v2 for LLM-judge calibration — informed by exp02 round 1 results.

Round 1 findings (bboxes only, 20 cases):
  pr_decomposed  R²= 0.50  ← WINNER: explicit P/R decomposition → F1 formula works
  soft_recall    R²= 0.20  ← recall emphasis has modest positive correlation
  domain_aware   R²=-0.06  ← domain context alone didn't help
  baseline       R²=-1.15  ← anchored strict-FP scale creates inverted correlation
  cot_counting   R²=-7.83  ← explicit object counting in UAV images is catastrophic

Design principles for v2
------------------------
1. Build on pr_decomposed (best): keep P/R split + formula, improve estimation guidance.
2. Avoid absolute counting (cot_counting failure): use proportions/fractions instead.
3. Avoid strict anchored scale (baseline failure): let model compute F1 mathematically.
4. Keep prompts short and focused — complex instructions degraded calibration.
5. Try hierarchical row-level reasoning as an alternative decomposition.
"""

_JSON_ONLY = (
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block."
)

# ---------------------------------------------------------------------------
# V2-1: PR_GUIDED
# Extended pr_decomposed with explicit visual cues for TP/FP/FN in crop images.
# Hypothesis: better estimation of P/R → higher R².
# ---------------------------------------------------------------------------
PR_GUIDED_SYSTEM = f"""
You are a Computer Vision evaluation expert for UAV agricultural imagery.
You estimate detection quality by visually separating correct detections from errors.

{_JSON_ONLY}
"""

PR_GUIDED_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

The attached UAV image shows model predictions in CYAN.

Estimate RECALL and PRECISION for the CYAN predictions:

RECALL = fraction of real {output_type} in the scene that are covered by a CYAN prediction.
  • A crop/row is "covered" when at least one CYAN prediction overlaps it or is very close.
  • Scan the entire image systematically — don't miss crops at the image borders.
  • Estimate: what % of the visible objects have a CYAN mark on them?

PRECISION = fraction of CYAN predictions that land on a real {output_type}.
  • A prediction is "wrong" if it falls on empty background, between rows, or on a non-crop.
  • Estimate: what % of CYAN marks are on actual crops/rows?

Then compute F1 = 2 × Recall × Precision / (Recall + Precision).

Return ONLY this JSON:
{{
  "recall": <float 0.0–1.0>,
  "precision": <float 0.0–1.0>,
  "metric_value": <float — F1 computed above>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# V2-2: PROPORTIONAL_PR
# Avoids counting; asks for proportions as percentages.
# Hypothesis: "what % is covered" is easier than counting and less prone to error.
# ---------------------------------------------------------------------------
PROPORTIONAL_PR_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You estimate detection coverage and noise as percentages, not absolute counts.

{_JSON_ONLY}
"""

PROPORTIONAL_PR_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

The attached image shows CYAN predictions on a UAV crop scene.

Answer two percentage questions:

COVERAGE %: Looking at all the real {output_type} visible in the scene,
what percentage of them have a CYAN prediction on or very near them?
(100% = every real object detected, 50% = half missed)

NOISE %: Looking at all the CYAN predictions,
what percentage are on empty background with NO real {output_type} nearby?
(0% = all predictions correct, 50% = half are spurious)

Then compute:
  recall    = COVERAGE% / 100
  precision = 1 - NOISE% / 100
  metric_value = 2 × recall × precision / (recall + precision)

Return ONLY this JSON:
{{
  "coverage_pct": <int 0–100>,
  "noise_pct": <int 0–100>,
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float F1>
}}
"""

# ---------------------------------------------------------------------------
# V2-3: ROW_COVERAGE
# Hierarchical: identify rows first, then check per-row coverage.
# Hypothesis: row-level reasoning reduces the count problem and uses domain structure.
# ---------------------------------------------------------------------------
ROW_COVERAGE_SYSTEM = f"""
You are a Computer Vision evaluation expert for precision agriculture.
You evaluate detection quality at the crop-row level before aggregating.

{_JSON_ONLY}
"""

ROW_COVERAGE_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

The attached UAV image shows CYAN predictions on a crop field.

Crops grow in parallel rows. Evaluate in two steps:

STEP 1 — Row inventory:
  Count the number of visible crop rows in the image.
  For each row, decide: does it have adequate CYAN prediction coverage? (yes/no)

STEP 2 — Within-row assessment (for rows that DO have predictions):
  Are there CYAN predictions between rows or in clearly empty areas (false positives)?

From this, estimate:
  row_recall    = rows_with_adequate_coverage / total_rows
  row_precision = (predictions on real crops) / (all predictions)
  metric_value  = 2 × row_recall × row_precision / (row_recall + row_precision)

Return ONLY this JSON:
{{
  "n_rows_visible": <int>,
  "n_rows_covered": <int>,
  "row_recall": <float 0.0–1.0>,
  "row_precision": <float 0.0–1.0>,
  "metric_value": <float F1>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# V2-4: ORDINAL_F1
# Avoids continuous estimation; first picks a coarse bucket, then refines.
# Hypothesis: reduces overconfidence; coarse → fine is more accurate than direct.
# ---------------------------------------------------------------------------
ORDINAL_F1_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You classify detection quality into coarse levels before refining to a continuous score.

{_JSON_ONLY}
"""

ORDINAL_F1_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

The attached image shows CYAN predictions on a UAV crop scene.

Step 1 — Pick the best matching quality level for the CYAN predictions:

  LEVEL A (F1 ~ 0.9–1.0): Nearly all real objects detected; very few wrong predictions.
  LEVEL B (F1 ~ 0.7–0.9): Most objects detected; some missed or some spurious.
  LEVEL C (F1 ~ 0.5–0.7): About half correct; noticeable gaps or noise.
  LEVEL D (F1 ~ 0.3–0.5): Majority missed or majority spurious.
  LEVEL E (F1 ~ 0.0–0.3): Very poor — almost nothing detected or nearly all wrong.

Step 2 — Within that level, estimate where exactly:
  e.g. LEVEL B closer to A (0.88) or closer to C (0.72)?

Set metric_value to your final continuous F1 estimate.

Return ONLY this JSON:
{{
  "level": "<A/B/C/D/E>",
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence: main strength and main weakness>"
}}
"""

# ---------------------------------------------------------------------------
# V2-5: SPLIT_ERROR_TYPES
# Explicitly separates missed detections (FN) from wrong detections (FP),
# then computes recall and precision from those fractions.
# Hypothesis: naming the two error types separately reduces conflation.
# ---------------------------------------------------------------------------
SPLIT_ERROR_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You diagnose detection errors by separating missed objects from spurious predictions.

{_JSON_ONLY}
"""

SPLIT_ERROR_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

The attached image shows CYAN predictions on a UAV crop scene.

Identify the two types of errors independently:

MISSED (false negatives):
  Look at each real {output_type} in the scene. Is there a CYAN prediction for it?
  Estimate the fraction of real objects that have NO CYAN prediction.
  → missed_fraction = FN / (TP + FN)   [0 = none missed, 1 = all missed]

WRONG (false positives):
  Look at each CYAN prediction. Does it land on a real {output_type}?
  Estimate the fraction of predictions that are spurious (no real object nearby).
  → wrong_fraction = FP / (TP + FP)   [0 = no wrong predictions, 1 = all wrong]

Then compute:
  recall    = 1 - missed_fraction
  precision = 1 - wrong_fraction
  metric_value = 2 × recall × precision / (recall + precision)

Return ONLY this JSON:
{{
  "missed_fraction": <float 0.0–1.0>,
  "wrong_fraction": <float 0.0–1.0>,
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float F1>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# Public dict consumed by run.py
# ---------------------------------------------------------------------------
PROMPTS_V2 = {
    "pr_guided": {
        "label": "PR Guided (visual cues)",
        "system": PR_GUIDED_SYSTEM,
        "user_template": PR_GUIDED_USER,
    },
    "proportional_pr": {
        "label": "Proportional % (no counting)",
        "system": PROPORTIONAL_PR_SYSTEM,
        "user_template": PROPORTIONAL_PR_USER,
    },
    "row_coverage": {
        "label": "Row-Level Coverage",
        "system": ROW_COVERAGE_SYSTEM,
        "user_template": ROW_COVERAGE_USER,
    },
    "ordinal_f1": {
        "label": "Ordinal Levels → Refine",
        "system": ORDINAL_F1_SYSTEM,
        "user_template": ORDINAL_F1_USER,
    },
    "split_errors": {
        "label": "Split Error Types (missed/wrong)",
        "system": SPLIT_ERROR_SYSTEM,
        "user_template": SPLIT_ERROR_USER,
    },
}
