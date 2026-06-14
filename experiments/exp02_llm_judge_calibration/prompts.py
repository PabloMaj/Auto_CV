"""
5 LLM-as-judge prompt variants for calibration experiment.

Each entry has:
  system       — system prompt
  user_template — user prompt with {user_prompt} and {output_type} placeholders

Prompt design rationale:
  1. baseline       — current production prompt (strict FP penalty, anchored scale)
  2. pr_decomposed  — explicit precision + recall → F1 formula
  3. cot_counting   — chain-of-thought: count GT / pred / TP before scoring
  4. domain_aware   — crop-row agricultural context
  5. soft_recall    — recall-dominant, moderate FP penalty
"""

_JSON_ONLY = (
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block."
)

# ---------------------------------------------------------------------------
# 1. BASELINE  (identical to current production prompt)
# ---------------------------------------------------------------------------
BASELINE_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You assess the quality of computer vision model predictions by visually inspecting images.

{_JSON_ONLY}
"""

BASELINE_USER = """
COMPUTER VISION TASK:
{user_prompt}

OUTPUT TYPE: {output_type}

The attached image shows model predictions drawn in CYAN.

Score the predictions on a scale 0.0–1.0. Be STRICT — default to a LOW score unless
predictions are clearly correct. Use this anchoring:

  1.0 — every real object detected, zero spurious predictions
  0.7 — most real objects detected, a few FP (spurious predictions)
  0.5 — roughly half correct, noticeable FP or FN
  0.3 — many FP or most real objects missed
  0.0 — completely wrong or nothing detected

FALSE POSITIVES ARE HEAVILY PENALISED: each prediction that does not correspond to a
real object in the scene should pull the score down significantly.
Count spurious predictions explicitly before scoring.

Return ONLY this JSON (no other text):
{{
  "metric_value": <float 0.0 to 1.0>,
  "reasoning": "<one sentence: N correct, M spurious, K missed>"
}}
"""

# ---------------------------------------------------------------------------
# 2. PR_DECOMPOSED  — estimate precision and recall separately, compute F1
# ---------------------------------------------------------------------------
PR_DECOMPOSED_SYSTEM = f"""
You are a Computer Vision evaluation expert specialising in precision-recall analysis.
You decompose detection quality into precision and recall before computing F1.

{_JSON_ONLY}
"""

PR_DECOMPOSED_USER = """
COMPUTER VISION TASK:
{user_prompt}

OUTPUT TYPE: {output_type}

The attached image shows model predictions drawn in CYAN.

Evaluate in two steps:

RECALL  = fraction of real objects in the scene that have a CYAN prediction nearby.
           (missed objects reduce recall)
PRECISION = fraction of CYAN predictions that correspond to a real object.
            (spurious / wrong predictions reduce precision)

Then compute:  F1 = 2 × Precision × Recall / (Precision + Recall)

Estimate Recall and Precision as decimals from 0.0 to 1.0 based on visual inspection.
Set metric_value = F1.

Return ONLY this JSON:
{{
  "precision": <float 0.0 to 1.0>,
  "recall": <float 0.0 to 1.0>,
  "metric_value": <float — F1 computed above>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# 3. COT_COUNTING  — explicit counting chain-of-thought before scoring
# ---------------------------------------------------------------------------
COT_COUNTING_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You always count objects carefully before computing any score.

{_JSON_ONLY}
"""

COT_COUNTING_USER = """
COMPUTER VISION TASK:
{user_prompt}

OUTPUT TYPE: {output_type}

The attached image shows model predictions drawn in CYAN.

Follow these steps explicitly:

Step 1 — Count all real {output_type} visible in the scene.  Call this N_gt.
Step 2 — Count all CYAN predictions in the image.           Call this N_pred.
Step 3 — Estimate how many CYAN predictions correctly match a real object (TP).
          A match means the prediction is spatially close to / overlapping a real object.
Step 4 — Derive:
            FP = N_pred - TP   (spurious, unmatched predictions)
            FN = N_gt  - TP    (missed real objects)
Step 5 — Compute F1 = 2·TP / (2·TP + FP + FN).
          Set metric_value = F1.

Return ONLY this JSON:
{{
  "n_gt_estimated": <int>,
  "n_pred": <int>,
  "tp_estimated": <int>,
  "fp_estimated": <int>,
  "fn_estimated": <int>,
  "metric_value": <float F1>,
  "reasoning": "<one sentence explaining your counts>"
}}
"""

# ---------------------------------------------------------------------------
# 4. DOMAIN_AWARE  — uses agricultural crop-row domain knowledge
# ---------------------------------------------------------------------------
DOMAIN_AWARE_SYSTEM = f"""
You are an expert in evaluating computer vision systems for precision agriculture.
You understand that crops in UAV images grow in structured, parallel rows with
approximately regular inter-plant and inter-row spacing.

{_JSON_ONLY}
"""

DOMAIN_AWARE_USER = """
COMPUTER VISION TASK:
{user_prompt}

OUTPUT TYPE: {output_type}

The attached image shows model predictions drawn in CYAN on a UAV crop image.

DOMAIN KNOWLEDGE:
- Crops grow in parallel rows; all rows share the same dominant orientation.
- Inter-plant spacing within a row is approximately uniform.
- Inter-row spacing is approximately constant across the image.
- Predictions that fall in background areas (between rows, outside crop rows) are spurious.
- A correct prediction matches a visible plant / crop row with good spatial agreement.

Evaluate quality considering:
1. Coverage — are all visible plants / rows represented by CYAN predictions?
2. Purity   — do CYAN predictions fall on real crops or on empty background / weeds?
3. Count    — does the number of predictions match the expected number of objects?

Score from 0.0 (completely wrong) to 1.0 (perfect detection of all crop objects).
Penalise both missed crops (FN) and predictions on empty areas (FP) equally.

Return ONLY this JSON:
{{
  "metric_value": <float 0.0 to 1.0>,
  "coverage_quality": "<all / most / partial / few>",
  "noise_level": "<none / low / moderate / high>",
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# 5. SOFT_RECALL  — recall-dominant; moderate FP penalty
# ---------------------------------------------------------------------------
SOFT_RECALL_SYSTEM = f"""
You are a Computer Vision evaluation expert.
You prioritise detection coverage (recall) as the primary quality signal,
with a moderate penalty for spurious predictions (false positives).

{_JSON_ONLY}
"""

SOFT_RECALL_USER = """
COMPUTER VISION TASK:
{user_prompt}

OUTPUT TYPE: {output_type}

The attached image shows model predictions drawn in CYAN.

Score predictions from 0.0 to 1.0 using these priorities:

PRIMARY (weight ~70 %):
  What fraction of real {output_type} visible in the image are detected by a CYAN prediction?
  Missing the majority of real objects is the worst failure mode.

SECONDARY (weight ~30 %):
  Are there too many spurious CYAN predictions with no real object nearby?
  A moderate number of extra detections is acceptable; extreme over-detection (3× more
  predictions than real objects) should reduce the score noticeably.

Anchoring:
  1.0 — all real objects detected, very few spurious predictions
  0.8 — >80 % detected, moderate spurious OK
  0.6 — ~60 % detected, some spurious
  0.4 — ~40–50 % detected or heavy spurious
  0.2 — <30 % detected
  0.0 — nothing detected or completely wrong

Return ONLY this JSON:
{{
  "metric_value": <float 0.0 to 1.0>,
  "reasoning": "<one sentence: coverage level and noise level>"
}}
"""

# ---------------------------------------------------------------------------
# Public dict consumed by run.py
# ---------------------------------------------------------------------------
PROMPTS = {
    "baseline": {
        "label": "Baseline (strict FP)",
        "system": BASELINE_SYSTEM,
        "user_template": BASELINE_USER,
    },
    "pr_decomposed": {
        "label": "P+R Decomposed → F1",
        "system": PR_DECOMPOSED_SYSTEM,
        "user_template": PR_DECOMPOSED_USER,
    },
    "cot_counting": {
        "label": "CoT Counting",
        "system": COT_COUNTING_SYSTEM,
        "user_template": COT_COUNTING_USER,
    },
    "domain_aware": {
        "label": "Domain-Aware (crop rows)",
        "system": DOMAIN_AWARE_SYSTEM,
        "user_template": DOMAIN_AWARE_USER,
    },
    "soft_recall": {
        "label": "Soft Recall-Dominant",
        "system": SOFT_RECALL_SYSTEM,
        "user_template": SOFT_RECALL_USER,
    },
}
