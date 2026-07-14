"""
Experiment 01: Multi-dataset benchmark

Runs the full pipeline for 4 datasets x 3 annotation variants (bbox, midpoints, lines).
Total: 12 runs. Each run gets its own exp_id and workspace directory.

Usage (from repo root):
    python experiments/exp01_multi_dataset.py
"""
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import SystemSettings
from src.graph.workflow import build_graph
from src.inference.sonnet_inference import reset_cost_tracker, save_cost_report
from src.state.agent_state import AgentState
from src.utils.state_utils import state_to_json

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

DATA_ROOT = "data/data_structured/crop_line_uav"

DATASETS = [
    "maize_3_nerac_2016_1",
    "sugarbeet_3_charmont_2017_1",
    "sunflower_1_auzeville_2019_1",
    "sunflower_3_riviere_2017_1",
]

_SCENE_CONTEXT = (
    "Crops are arranged in parallel rows with approximately equal and constant spacing between them. "
    "The spacing between adjacent rows is approximately uniform across the entire image. "
    "The centers of individual plants lie approximately on straight crop row lines."
)

VARIANTS = {
    "bboxes": {
        "eval_suffix": "bboxes",
        "user_prompt": (
            "Develop a computer vision method to detect crops in given RGB images. "
            "You can use only classical computer vision techniques for the solution. Return bounding boxes."
        ),
    },
    "midpoints": {
        "eval_suffix": "midpoints",
        "user_prompt": (
            "Develop a computer vision method to detect crops in given RGB images. "
            "You can use only classical computer vision techniques for the solution. Return midpoints."
        ),
    },
    "lines": {
        "eval_suffix": "lines",
        "user_prompt": (
            "Develop a computer vision method to detect crop row lines in given RGB images. "
            f"{_SCENE_CONTEXT} "
            "Each detected line segment should approximately start and end at the outermost plants of the crop row "
            "and should span most of the image. "
            "Return exactly one line segment per crop row — do not split a single row into multiple short segments, "
            "as short fragments belonging to the same row will be counted as false positives. "
            "For example, if two crop rows are visible in the image, return exactly two line segments. "
            "Not all detected objects need to belong to a crop row — some detections may represent weeds or noise. "
            "The algorithm should distinguish between crop row plants and outlier detections. "
            "A minimum number of plants should be required to form a valid crop row (e.g. at least 4–5 aligned detections); "
            "clusters with fewer detections should be discarded rather than returned as a line. "
            "Prefer long line segments: a line supported by many inlier plants is more reliable than a short one. "
            "Note that significant gaps between plants within a row may occur; the method should be robust to missing plants "
            "and must not split a row into separate segments just because of a gap. "
            "All detected line segments should have approximately the same orientation. "
            "Estimating this dominant orientation from the data can be an explicit step in the processing pipeline. "
            "The maximum allowed angular deviation of any returned line segment from the dominant crop row orientation is 10 degrees. "
            "Line segments that deviate by more than 10 degrees from the dominant orientation are considered incorrect detections "
            "and will be penalised in the metric as false positives. "
            "You can use only classical computer vision techniques for the solution. "
            "Return line segments."
        ),
    },
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(settings: SystemSettings, dataset_name: str, variant_name: str, variant_cfg: dict) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp}_{dataset_name}_{variant_name}"

    dl_dataset_path = f"{DATA_ROOT}/{dataset_name}_bboxes"
    eval_dataset_path = f"{DATA_ROOT}/{dataset_name}_{variant_cfg['eval_suffix']}"

    initial_state = AgentState(
        user_prompt=variant_cfg["user_prompt"],
        dl_dataset_path=dl_dataset_path,
        eval_dataset_path=eval_dataset_path,
        exp_id=exp_id,
    )

    reset_cost_tracker()
    graph = build_graph(settings)
    result = graph.invoke(initial_state.model_dump())

    exp_dir = Path("workspace") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "final_state.json", "w", encoding="utf-8") as f:
        json.dump(state_to_json(result), f, indent=2, ensure_ascii=False)

    with open(exp_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2, ensure_ascii=False)

    save_cost_report(exp_dir / "cost_report.json")

    return {"exp_id": exp_id, "status": "success"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = SystemSettings()

    runs = [
        (dataset, variant_name, variant_cfg)
        for dataset in DATASETS
        for variant_name, variant_cfg in VARIANTS.items()
        if variant_name in ["bboxes", "midpoints", "lines"]
    ]

    print(f"Experiment 01 — {len(runs)} runs total")
    print(f"Datasets : {DATASETS}")
    print(f"Variants : {list(VARIANTS.keys())}")

    summary = []

    for i, (dataset, variant_name, variant_cfg) in enumerate(runs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(runs)}] {dataset}  [{variant_name}]")
        print(f"{'=' * 60}")

        try:
            run_result = run_single(settings, dataset, variant_name, variant_cfg)
            summary.append({
                "run": i,
                "dataset": dataset,
                "variant": variant_name,
                "exp_id": run_result["exp_id"],
                "status": "success",
            })
            print(f"Done → workspace/{run_result['exp_id']}")

        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()
            summary.append({
                "run": i,
                "dataset": dataset,
                "variant": variant_name,
                "status": "failed",
                "error": str(exc),
            })

    # Save experiment-level summary
    summary_path = Path("workspace") / f"exp01_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete  —  summary: {summary_path}")
    print(f"{'=' * 60}")
    for s in summary:
        tag = "OK " if s["status"] == "success" else "ERR"
        exp_ref = s.get("exp_id", s.get("error", ""))
        print(f"  [{tag}] {s['dataset']:45s} [{s['variant']:10s}]  {exp_ref}")
