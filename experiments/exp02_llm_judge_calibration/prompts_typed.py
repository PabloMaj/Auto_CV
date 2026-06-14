"""
Typed prompt variants — structural adaptations of pr_f1_definition (V5-4, best R²≈0.59)
for each output type.

Design principle
----------------
Templates describe the annotation format (rectangles / center points / line segments)
but contain zero domain knowledge — no mention of plants, crop rows, UAV, agriculture, etc.
All domain context comes exclusively from {user_prompt}.

Variants differ in annotation format description and matching framing:
  typed_bboxes    — CYAN rectangles, overlap-based matching
  typed_midpoints — CYAN center points, proximity-based matching
  typed_lines     — CYAN line segments, coverage-based matching (one per structure)
"""

_JSON_ONLY = (
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON block."
)

_SYSTEM = f"""
You are a Computer Vision evaluation expert.
{_JSON_ONLY}
"""

# ---------------------------------------------------------------------------
# typed_bboxes
# Annotation format: rectangular bounding boxes.
# ---------------------------------------------------------------------------
_BBOXES_USER = """
TASK: {user_prompt}

CYAN rectangular bounding boxes are drawn around predicted objects in the attached image.

F1 score measures detection quality by balancing two aspects:
  — recall:    what fraction of real objects have a CYAN bounding box on them
  — precision: what fraction of CYAN bounding boxes are on a real object
F1 is low if either aspect is poor; F1 is high only when both are good.

Estimate:
  recall:       fraction of real objects with a CYAN bounding box (0–1)
  precision:    fraction of CYAN bounding boxes on a real object (0–1)
  metric_value: the F1 score (0 = worst, 1 = perfect)

Return ONLY this JSON:
{{
  "recall": <float 0.0–1.0>,
  "precision": <float 0.0–1.0>,
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# typed_midpoints
# Annotation format: center point markers.
# ---------------------------------------------------------------------------
_MIDPOINTS_USER = """
TASK: {user_prompt}

CYAN dots mark predicted object center positions in the attached image.

F1 score measures detection quality by balancing two aspects:
  — recall:    what fraction of real objects have a CYAN dot on or near their center
  — precision: what fraction of CYAN dots are on or near the center of a real object
F1 is low if either aspect is poor; F1 is high only when both are good.

Estimate:
  recall:       fraction of real objects with a nearby CYAN center dot (0–1)
  precision:    fraction of CYAN dots that are near a real object center (0–1)
  metric_value: the F1 score (0 = worst, 1 = perfect)

Return ONLY this JSON:
{{
  "recall": <float 0.0–1.0>,
  "precision": <float 0.0–1.0>,
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# typed_lines
# Annotation format: line segments, one per structure.
# ---------------------------------------------------------------------------
_LINES_USER = """
TASK: {user_prompt}

CYAN line segments are drawn along predicted structures in the attached image.
The expected output is exactly one CYAN segment per real structure.

F1 score measures detection quality by balancing two aspects:
  — recall:    what fraction of real structures are covered by a CYAN segment
  — precision: what fraction of CYAN segments correspond to a distinct real structure
    (a redundant segment on an already-covered structure counts as incorrect)
F1 is low if either aspect is poor; F1 is high only when both are good.

Estimate:
  recall:       fraction of real structures covered by a CYAN segment (0–1)
  precision:    fraction of CYAN segments corresponding to a distinct real structure (0–1)
  metric_value: the F1 score (0 = worst, 1 = perfect)

Return ONLY this JSON:
{{
  "recall": <float 0.0–1.0>,
  "precision": <float 0.0–1.0>,
  "metric_value": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# Public dict — variant_only ensures each prompt runs only on its variant
# ---------------------------------------------------------------------------
PROMPTS_TYPED = {
    "typed_bboxes": {
        "label": "Typed bboxes (rectangular boxes)",
        "system": _SYSTEM,
        "user_template": _BBOXES_USER,
        "variant_only": "bboxes",
    },
    "typed_midpoints": {
        "label": "Typed midpoints (center dots)",
        "system": _SYSTEM,
        "user_template": _MIDPOINTS_USER,
        "variant_only": "midpoints",
    },
    "typed_lines": {
        "label": "Typed lines (one segment per structure)",
        "system": _SYSTEM,
        "user_template": _LINES_USER,
        "variant_only": "lines",
    },
}
