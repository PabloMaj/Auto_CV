"""
Prompt variants v5 — informed by rounds 1–4.

V4 key findings
---------------
  pr_no_formula       R²=+0.591  ← NEW BEST: removing F1 formula improves calibration
  pr_evidence_first   R²=+0.463  ← observation step helps
  pr_range_hint       R²=-0.667  ← "use full range" causes over-correction
  pr_with_npred       R²=-3.419  ← model sees n_pred=250, infers "good coverage" → wrong
  pr_strict_decimals  R²=-5.403  ← "no rounding" → model defaults to 0.0 when uncertain

Full history of best R² per round:
  v1 pr_decomposed  +0.50
  v2 ordinal_f1     +0.58
  v3 pr_minimal     +0.51
  v4 pr_no_formula  +0.591  ← current best

Why pr_no_formula wins
----------------------
Showing "metric_value = 2×P×R/(P+R)" causes the model to:
  (a) anchor to formula midpoints during computation
  (b) make arithmetic errors that compound
Without the formula, the model uses its pre-trained F1 intuition — more accurate.

Why pr_evidence_first helps
---------------------------
Forcing an observation sentence before the estimate commits the model to
visual evidence, reducing anchoring to a safe middle-ground value.

Design goals for v5
--------------------
1. PRIMARY: combine pr_no_formula + observation step → best of both.
2. Test: ask for F1 DIRECTLY without any P/R decomposition.
3. Test: ask P/R as percentages (0–100 integers) — may be more natural to estimate.
4. Test: provide F1 definition (not formula) before asking — inform without anchoring.
5. AVOID: categorical levels, formula derivation, n_pred, "use full range" instruction.
"""

_JSON_ONLY = (
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block."
)

# ---------------------------------------------------------------------------
# V5-1: PR_EVIDENCE_NO_FORMULA  ← PRIMARY HYPOTHESIS
# Combines the two best v4 prompts:
#   pr_evidence_first (R²=0.46): observation before estimation
#   pr_no_formula (R²=0.59):     no formula derivation in prompt
# Hypothesis: committing to visual evidence + using pre-trained F1 sense = best combo.
# ---------------------------------------------------------------------------
PR_EVIDENCE_NO_FORMULA_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_EVIDENCE_NO_FORMULA_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions are drawn on the attached UAV crop image.

Step 1 — Observe: briefly describe (a) any visible {output_type} that have NO CYAN mark,
and (b) any CYAN marks that fall on empty background with no {output_type} nearby.

Step 2 — Estimate:
  recall:       what fraction of real {output_type} have a CYAN prediction on them?
  precision:    what fraction of CYAN predictions are on a real {output_type}?
  metric_value: the F1 score (harmonic mean of recall and precision)

Return ONLY:
{{
  "observation": "<one sentence: what errors you see>",
  "recall": <float 0.0–1.0>,
  "precision": <float 0.0–1.0>,
  "metric_value": <float 0.0–1.0>
}}
"""

# ---------------------------------------------------------------------------
# V5-2: DIRECT_F1
# Ask for F1 directly — NO P/R decomposition at all.
# Hypothesis: decomposing into P and R may introduce compounding estimation errors.
# A single holistic F1 estimate (informed by model's pre-trained understanding) may
# outperform the two-step P/R path.
# ---------------------------------------------------------------------------
DIRECT_F1_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

DIRECT_F1_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions are drawn on the attached UAV crop image.

Estimate the F1 score for the CYAN predictions.
F1 = 0 when nothing is detected correctly; F1 = 1 when every real {output_type}
is detected and every CYAN prediction is on a real {output_type}.

Return ONLY:
{{
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# V5-3: PR_PERCENTAGE
# Ask recall and precision as integers 0–100 instead of floats 0–1.
# Hypothesis: "what % of plants are marked?" is more natural to estimate
# than "what fraction of plants are marked?", and avoids the 0.64/0.76/0.78
# clustering observed in pr_minimal.
# No formula shown — consistent with v4 winner.
# ---------------------------------------------------------------------------
PR_PERCENTAGE_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_PERCENTAGE_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV crop image.

Estimate as whole percentages:
  recall_pct:    of all real {output_type} visible, what % have a CYAN mark on them?
  precision_pct: of all CYAN marks visible, what % are on a real {output_type}?
  metric_value:  the F1 score corresponding to these two percentages

Return ONLY:
{{
  "recall_pct": <int 0–100>,
  "precision_pct": <int 0–100>,
  "metric_value": <float 0.0–1.0>
}}
"""

# ---------------------------------------------------------------------------
# V5-4: PR_F1_DEFINITION
# Provide a plain-English definition of F1 BEFORE asking for P/R.
# Tests whether informing the model about what F1 means (without showing
# the arithmetic formula) improves calibration of metric_value.
# Distinct from pr_no_formula (no definition) and pr_with_formula (formula shown).
# ---------------------------------------------------------------------------
PR_F1_DEFINITION_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_F1_DEFINITION_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV crop image.

F1 score measures detection quality by balancing two aspects:
  — recall:    how many real {output_type} are covered by a CYAN prediction
  — precision: how many CYAN predictions land on a real {output_type}
F1 is low if either aspect is poor; F1 is high only when both are good.

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

# ---------------------------------------------------------------------------
# V5-5: PR_EVIDENCE_PERCENTAGE
# Combines observation step + percentage framing (both untested together).
# Hypothesis: evidence commitment + integer scale = reduces both anchoring
# and clustering problems simultaneously.
# ---------------------------------------------------------------------------
PR_EVIDENCE_PERCENTAGE_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_EVIDENCE_PERCENTAGE_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV crop image.

Step 1 — Observe: briefly note (a) {output_type} visible without any CYAN mark,
and (b) CYAN marks that appear to be on empty background.

Step 2 — Estimate as whole percentages:
  recall_pct:    of all real {output_type}, what % have a CYAN mark on them?
  precision_pct: of all CYAN marks, what % are on a real {output_type}?
  metric_value:  the F1 score for these values (0 = worst, 1 = perfect)

Return ONLY:
{{
  "observation": "<one sentence>",
  "recall_pct": <int 0–100>,
  "precision_pct": <int 0–100>,
  "metric_value": <float 0.0–1.0>
}}
"""

# ---------------------------------------------------------------------------
# Public dict consumed by run.py
# ---------------------------------------------------------------------------
PROMPTS_V5 = {
    "pr_evidence_no_formula": {
        "label": "Evidence + No Formula (v4 combo)",
        "system": PR_EVIDENCE_NO_FORMULA_SYSTEM,
        "user_template": PR_EVIDENCE_NO_FORMULA_USER,
    },
    "direct_f1": {
        "label": "Direct F1 (no P/R split)",
        "system": DIRECT_F1_SYSTEM,
        "user_template": DIRECT_F1_USER,
    },
    "pr_percentage": {
        "label": "P/R as % integers, no formula",
        "system": PR_PERCENTAGE_SYSTEM,
        "user_template": PR_PERCENTAGE_USER,
    },
    "pr_f1_definition": {
        "label": "P/R + F1 definition (no formula)",
        "system": PR_F1_DEFINITION_SYSTEM,
        "user_template": PR_F1_DEFINITION_USER,
    },
    "pr_evidence_percentage": {
        "label": "Evidence + % integers",
        "system": PR_EVIDENCE_PERCENTAGE_SYSTEM,
        "user_template": PR_EVIDENCE_PERCENTAGE_USER,
    },
}
