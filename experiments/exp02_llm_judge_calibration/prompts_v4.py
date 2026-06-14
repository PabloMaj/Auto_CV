"""
Prompt variants v4 — informed by rounds 1, 2, 3.

Key diagnosis from v3
---------------------
Every categorical / ordinal prompt collapses to one or two "parking-spot" values:
  ordinal_visual  → 13/20 scores = 0.58 (midpoint of Level C)
  dual_ordinal    → 15/20 scores = 0.6346 (default R-GOOD × P-MED combo)
  holistic_cal.   → 6/20 at 0.45

When the model is uncertain about a UAV image (tiny plants from altitude),
it selects the most defensible middle category and outputs its midpoint.
No level description is good enough to escape this attractor.

pr_minimal (R²=0.51) works BEST because writing the formula forces commitment
to concrete recall and precision values — but the spread is still limited (0.50–0.80).

Design goals for v4
--------------------
1. ZERO categorical levels — no A/B/C/D/E, no HIGH/MED/LOW. Pure continuous.
2. Ground the estimate in a known anchor: n_pred (always available in deployment).
3. Force observation BEFORE estimation (commits model to visual evidence).
4. Anti-anchoring: tell model scores span 0.4–1.0, don't default to middle.
5. Test removing the formula (does formula-writing help or hurt?).
6. Force non-rounded decimals to increase spread.

History summary
---------------
v1 pr_decomposed   R²=+0.50   P/R decomposition + formula
v2 ordinal_f1      R²=+0.58   coarse bucket → refine  ← overall best so far
v2 proportional_pr R²=+0.41   coverage% + noise% estimation
v3 pr_minimal      R²=+0.51   simplest P/R formula (3 lines)
v3 dual_ordinal    R²=-0.12   ordinal for P and R separately
v3 ordinal_visual  R²=+0.14   ordinal with visual UAV anchors
v1 split_errors    R²=-0.67   missed/wrong fraction framing
v1 cot_counting    R²=-7.83   explicit counting is catastrophic
"""

_JSON_ONLY = (
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block."
)

# ---------------------------------------------------------------------------
# V4-1: PR_WITH_NPRED
# Give the model the exact number of CYAN predictions as an anchor.
# With n_pred=250 and ~78 visible plants, the model can reason:
#   "250 predictions for ~80 plants is 3× coverage → precision is low"
# n_pred is always available in deployment (it is the algorithm output count).
# ---------------------------------------------------------------------------
PR_WITH_NPRED_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_WITH_NPRED_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

The UAV image shows exactly {n_pred} CYAN predictions.

Estimate:
  recall    = fraction of real {output_type} that have a CYAN prediction on them (0–1)
  precision = fraction of the {n_pred} CYAN predictions that land on a real {output_type} (0–1)
  metric_value = 2 × recall × precision / (recall + precision)

Return ONLY:
{{
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float>
}}
"""

# ---------------------------------------------------------------------------
# V4-2: PR_EVIDENCE_FIRST
# Force a short visual observation BEFORE the P/R estimate.
# Committing to observations (e.g., "large empty area in top-right has cyan marks")
# should push the model away from the safe middle-ground estimate.
# ---------------------------------------------------------------------------
PR_EVIDENCE_FIRST_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_EVIDENCE_FIRST_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions are drawn on the attached UAV crop image.

Step 1 — Observe: note briefly (a) areas where visible {output_type} have NO CYAN mark,
and (b) areas where CYAN marks fall on empty background with no {output_type}.

Step 2 — Estimate:
  recall    = fraction of real {output_type} with a CYAN prediction (0–1)
  precision = fraction of CYAN predictions on a real {output_type} (0–1)
  metric_value = 2 × recall × precision / (recall + precision)

Return ONLY:
{{
  "observation": "<one sentence describing missed objects or spurious predictions you see>",
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float>
}}
"""

# ---------------------------------------------------------------------------
# V4-3: PR_RANGE_HINT
# Minimal change to pr_minimal: add explicit anti-anchoring note.
# Test whether simply saying "scores span 0.4–1.0, don't default to middle"
# is enough to push the model off the 0.60–0.70 attractor.
# ---------------------------------------------------------------------------
PR_RANGE_HINT_SYSTEM = f"""
You are a Computer Vision evaluation expert.
Detection quality in this experiment spans from poor (F1 ≈ 0.4, many objects missed
or many predictions spurious) to near-perfect (F1 ≈ 1.0, everything correct).
Some images will be at the extremes — do NOT default to a moderate middle estimate.
{_JSON_ONLY}
"""

PR_RANGE_HINT_USER = """
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
# V4-4: PR_NO_FORMULA
# Ask for recall, precision, and metric_value without showing the F1 formula.
# Tests whether showing the formula helps (forces mental computation) or hurts
# (anchors model to formula midpoints).
# ---------------------------------------------------------------------------
PR_NO_FORMULA_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_NO_FORMULA_USER = """
TASK: {user_prompt}
OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV image.

Estimate the detection quality:
  recall:         what fraction of real {output_type} have a CYAN prediction on them?
  precision:      what fraction of CYAN predictions are on a real {output_type}?
  metric_value:   the F1 score (harmonic mean of recall and precision)

Return ONLY:
{{
  "recall": <float 0.0–1.0>,
  "precision": <float 0.0–1.0>,
  "metric_value": <float 0.0–1.0>
}}
"""

# ---------------------------------------------------------------------------
# V4-5: PR_STRICT_DECIMALS
# Identical task to pr_minimal but explicitly requires non-rounded decimals.
# Addresses the clustering problem: scores clustered at 0.64, 0.76, 0.78.
# Test whether the instruction "use at least 2 significant figures, do not round
# to nearest 0.05 or 0.1" increases the variance of predictions.
# ---------------------------------------------------------------------------
PR_STRICT_DECIMALS_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

PR_STRICT_DECIMALS_USER = """
TASK: {user_prompt}  OUTPUT TYPE: {output_type}

CYAN predictions on the attached UAV image.

Estimate:
  recall    = fraction of real {output_type} that have a CYAN prediction on them  (0–1)
  precision = fraction of CYAN predictions that land on a real {output_type}        (0–1)
  metric_value = 2 × recall × precision / (recall + precision)

IMPORTANT: express recall and precision as precise decimals with at least 2 significant
figures after the decimal point (e.g., 0.63, 0.81, 0.47 — not 0.6, 0.8, 0.5).
Do NOT round to the nearest 0.05 or 0.10.

Return ONLY:
{{
  "recall": <float>,
  "precision": <float>,
  "metric_value": <float>
}}
"""

# ---------------------------------------------------------------------------
# Public dict consumed by run.py
# ---------------------------------------------------------------------------
PROMPTS_V4 = {
    "pr_with_npred": {
        "label": "P/R with n_pred anchor",
        "system": PR_WITH_NPRED_SYSTEM,
        "user_template": PR_WITH_NPRED_USER,
        "needs_npred": True,
    },
    "pr_evidence_first": {
        "label": "P/R with observation first",
        "system": PR_EVIDENCE_FIRST_SYSTEM,
        "user_template": PR_EVIDENCE_FIRST_USER,
    },
    "pr_range_hint": {
        "label": "P/R with anti-anchoring hint",
        "system": PR_RANGE_HINT_SYSTEM,
        "user_template": PR_RANGE_HINT_USER,
    },
    "pr_no_formula": {
        "label": "P/R without formula",
        "system": PR_NO_FORMULA_SYSTEM,
        "user_template": PR_NO_FORMULA_USER,
    },
    "pr_strict_decimals": {
        "label": "P/R with forced precision",
        "system": PR_STRICT_DECIMALS_SYSTEM,
        "user_template": PR_STRICT_DECIMALS_USER,
    },
}
