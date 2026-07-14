"""
Shared config for Experiment 04 — Dataset Enricher (pseudo-labelling) ablation.

Imported by both run.py (training) and summarize.py (aggregation) so the
directory layout only needs to be defined in one place.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = _REPO_ROOT / "workspace" / "publication" / "supervised" / "bboxes"
OUT_DIR = _REPO_ROOT / "workspace" / "exp04_dataset_enricher_eval"

# 4 saved publication examples, each holding a tiled_dataset (A) and a
# final_dataset (B, pseudo-label enriched) variant under the same
# dataset_enrichment_pseudo/crop_line_uav/<name>_bboxes directory.
DATASETS = [
    {"name": "sunflower_3_riviere_2017_1", "exp_dir": "20260608_232957_sunflower_3_riviere_2017_1_bboxes"},
    {"name": "sunflower_1_auzeville_2019_1", "exp_dir": "20260611_214019_sunflower_1_auzeville_2019_1_bboxes"},
    {"name": "sugarbeet_3_charmont_2017_1", "exp_dir": "20260611_224003_sugarbeet_3_charmont_2017_1_bboxes"},
    {"name": "maize_3_nerac_2016_1", "exp_dir": "20260611_233906_maize_3_nerac_2016_1_bboxes"},
]

# variant_key -> subfolder name under the dataset dir
VARIANTS = {
    "A_no_enrichment": "tiled_dataset",
    "B_pseudo_enrichment": "final_dataset",
}

N_REPEATS = 3

# Training hyperparameters — kept identical across datasets/variants/repeats,
# matching the defaults used by YOLOPipeline / SystemSettings.yolo_model_weights.
MODEL_WEIGHTS = "yolo11n.pt"
TRAIN_KWARGS = {
    "epochs": 200,
    "imgsz": 640,
    "batch": 16,
}


def dataset_root(dataset: dict) -> Path:
    return (
        BASE_DIR / dataset["exp_dir"] / "dataset_enrichment_pseudo" / "crop_line_uav"
        / f"{dataset['name']}_bboxes"
    )


def run_dir(dataset_name: str, variant_key: str, repeat: int) -> Path:
    return OUT_DIR / dataset_name / variant_key / f"run_{repeat}"
