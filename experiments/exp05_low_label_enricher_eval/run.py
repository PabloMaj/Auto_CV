"""
Experiment 05 — low-label-budget Dataset Enricher ablation

Trains YOLO on the datasets built by build_datasets.py:

  A_limited_no_enrichment     — only a 20% labeled subset (train+val)
  B_limited_pseudo_enrichment — same 20% labeled subset + full pseudo-label pool in train

Both variants are evaluated on the same full, original test split. Same
training hyperparameters (imported from exp04) are used for every run; each
(dataset, variant) combination is repeated 3x with a different seed.
Total: 4 datasets x 2 variants x 3 repeats = 24 YOLO trainings.

Usage (from repo root — build the datasets first):
    python experiments/exp05_low_label_enricher_eval/build_datasets.py
    python experiments/exp05_low_label_enricher_eval/run.py

Then summarize with:
    python experiments/exp05_low_label_enricher_eval/summarize.py

Set EXP05_TARGET_OBJECTS / EXP05_OBJECT_TOLERANCE (must match the values used
for build_datasets.py) to target a non-default object budget, e.g.:
    EXP05_TARGET_OBJECTS=20 EXP05_OBJECT_TOLERANCE=5 python experiments/exp05_low_label_enricher_eval/run.py
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp05_low_label_enricher_eval.config import (
    DATASETS, VARIANTS, N_REPEATS, MODEL_WEIGHTS, TRAIN_KWARGS, OUT_DIR,
    variant_dataset_dir, run_dir,
)
from src.funcs.dl_model_trainer_funcs.yolo_trainer import YOLOTrainer
from src.funcs.dl_model_trainer_funcs.yolo_evaluator import YOLOEvaluator


def run_single(yaml_path: Path, out_dir: Path, seed: int) -> dict:
    trainer = YOLOTrainer(model_weights=MODEL_WEIGHTS, data_yaml=str(yaml_path), output_dir=out_dir)
    trainer.train(seed=seed, save=True, **TRAIN_KWARGS)

    best_weights = out_dir / "weights" / "best.pt"
    evaluator = YOLOEvaluator(weights_path=str(best_weights), data_yaml=str(yaml_path), output_dir=out_dir)
    test_results = evaluator.test()

    return getattr(test_results, "results_dict", {}) or {}


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(DATASETS) * len(VARIANTS) * N_REPEATS
    print(f"Experiment 05 — low-label-budget Dataset Enricher ablation — {total_runs} runs total")
    print(f"Datasets ({len(DATASETS)}): {[d['name'] for d in DATASETS]}")
    print(f"Variants: {list(VARIANTS.keys())}")
    print(f"Repeats per (dataset, variant): {N_REPEATS}")
    print(f"Train kwargs: {TRAIN_KWARGS}  model={MODEL_WEIGHTS}")

    manifest = []
    run_idx = 0

    for dataset in DATASETS:
        for variant_key in VARIANTS:
            data_dir = variant_dataset_dir(dataset["name"], variant_key)
            yaml_path = data_dir / "data.yaml"

            if not yaml_path.exists():
                print(f"[SKIP] missing data.yaml: {yaml_path}  (run build_datasets.py first)")
                continue

            for repeat in range(1, N_REPEATS + 1):
                run_idx += 1
                out_dir = run_dir(dataset["name"], variant_key, repeat)
                seed = 1000 + repeat

                print(f"\n{'=' * 60}")
                print(f"[{run_idx}/{total_runs}] {dataset['name']}  [{variant_key}]  repeat {repeat}  seed={seed}")
                print(f"{'=' * 60}")

                entry = {
                    "run": run_idx,
                    "dataset": dataset["name"],
                    "variant": variant_key,
                    "repeat": repeat,
                    "seed": seed,
                    "run_dir": str(out_dir),
                }

                try:
                    metrics = run_single(yaml_path, out_dir, seed)
                    entry["status"] = "success"
                    entry["test_metrics"] = metrics
                    print(f"Done -> {out_dir}")

                except Exception as exc:
                    print(f"FAILED: {exc}")
                    traceback.print_exc()
                    entry["status"] = "failed"
                    entry["error"] = str(exc)

                manifest.append(entry)

                manifest_path = OUT_DIR / "manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete — manifest: {OUT_DIR / 'manifest.json'}")
    print(f"{'=' * 60}")
    for e in manifest:
        tag = "OK " if e["status"] == "success" else "ERR"
        print(f"  [{tag}] {e['dataset']:32s} [{e['variant']:28s}] repeat {e['repeat']}  {e['run_dir']}")

    print("\nNext: python experiments/exp05_low_label_enricher_eval/summarize.py")
