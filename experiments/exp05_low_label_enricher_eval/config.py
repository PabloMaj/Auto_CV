"""
Shared config for Experiment 05 — low-label-budget Dataset Enricher ablation.

Same 4 datasets / 3 repeats / 2 modes design as exp04, but the "labeled" pool
is artificially shrunk to a small object-count budget: tiles are randomly
selected from the original tiled_dataset train/val splits until the total
number of annotated objects reaches TARGET_OBJECTS +/- OBJECT_TOLERANCE
(applied independently to train and to val), to test whether pseudo-label
enrichment can compensate for a small labeled budget.

  A_limited_no_enrichment    — only the object-count-budgeted labeled subset, no enrichment
  B_limited_pseudo_enrichment — same labeled subset + full pseudo-label pool merged into train

Model/training hyperparameters (MODEL_WEIGHTS, TRAIN_KWARGS) and the dataset
list are imported from exp04's config so the two experiments stay in sync
(e.g. if epochs is tuned for exp04, exp05 automatically follows).

TARGET_OBJECTS / OBJECT_TOLERANCE are overridable via env vars so different
budgets can be built/trained/summarized side by side without overwriting
each other — the budget is baked into OUT_DIR, so each budget gets its own
datasets/ and run/ subtree:

    EXP05_TARGET_OBJECTS=20 EXP05_OBJECT_TOLERANCE=5 python experiments/exp05_low_label_enricher_eval/build_datasets.py
    EXP05_TARGET_OBJECTS=20 EXP05_OBJECT_TOLERANCE=5 python experiments/exp05_low_label_enricher_eval/run.py
    EXP05_TARGET_OBJECTS=20 EXP05_OBJECT_TOLERANCE=5 python experiments/exp05_low_label_enricher_eval/summarize.py
"""

import json
import os
from pathlib import Path

from experiments.exp04_dataset_enricher_eval.config import (
    DATASETS, MODEL_WEIGHTS, TRAIN_KWARGS, N_REPEATS,
    dataset_root as source_dataset_root,
)

# Datasets whose pseudo-label prompt optimization (dataset_enricher's
# prompt_opt/history.json) never reached this AP50 are excluded from the
# summarize.py report — their pseudo-labels are considered too noisy to draw
# an enrichment-vs-no-enrichment conclusion from.
AP50_QUALITY_THRESHOLD = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Target total number of annotated objects in the labeled train subset (and,
# independently, in the val subset) — NOT a fraction of tiles. Tiles are
# picked randomly (DATA_SELECTION_SEED) and accepted/rejected to land the
# cumulative object count within +/- OBJECT_TOLERANCE of TARGET_OBJECTS.
# Reused identically across both variants and all repeats (only the YOLO
# training seed varies across repeats, exactly like exp04).
TARGET_OBJECTS = int(os.environ.get("EXP05_TARGET_OBJECTS", "20"))
OBJECT_TOLERANCE = int(os.environ.get("EXP05_OBJECT_TOLERANCE", "5"))
DATA_SELECTION_SEED = 42
N_SELECTION_TRIALS = 2000  # random restarts to get close to TARGET_OBJECTS

_BUDGET_TAG = f"obj_{TARGET_OBJECTS}pm{OBJECT_TOLERANCE}"  # e.g. obj_20pm5

OUT_DIR = _REPO_ROOT / "workspace" / "exp05_low_label_enricher_eval" / _BUDGET_TAG
DATASETS_OUT_DIR = OUT_DIR / "datasets"

VARIANTS = {
    "A_limited_no_enrichment": "limited_dataset",
    "B_limited_pseudo_enrichment": "limited_enriched_dataset",
}


def dataset_out_dir(dataset_name: str) -> Path:
    return DATASETS_OUT_DIR / dataset_name


def variant_dataset_dir(dataset_name: str, variant_key: str) -> Path:
    return dataset_out_dir(dataset_name) / VARIANTS[variant_key]


def run_dir(dataset_name: str, variant_key: str, repeat: int) -> Path:
    return OUT_DIR / dataset_name / variant_key / f"run_{repeat}"


def prompt_opt_history_path(dataset: dict) -> Path:
    return source_dataset_root(dataset) / "prompt_opt" / "history.json"


def max_prompt_opt_ap50(dataset: dict):
    """Best AP50 ever reached during the dataset's pseudo-label prompt
    optimization, or None if no history.json is found."""
    path = prompt_opt_history_path(dataset)
    if not path.exists():
        return None

    entries = json.loads(path.read_text(encoding="utf-8"))
    values = [e.get("AP50", 0.0) for e in entries]
    return max(values) if values else None
