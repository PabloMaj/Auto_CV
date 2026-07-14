"""
Experiment 04 — Dataset Enricher (pseudo-labelling) ablation

Compares two YOLO dataset-preparation variants using the 4 saved publication
examples under workspace/publication/supervised/bboxes:

  A) tiled_dataset   — no enrichment (raw split-safe tiles only)
  B) final_dataset   — dataset-enrichment pseudo-labelling merged into train

Val/test splits are identical between A and B for a given dataset — only the
train split differs (B has extra pseudo-labeled train tiles) — so evaluating
both on the same held-out test split is a fair comparison.

Same training hyperparameters are used for every run; each (dataset, variant)
combination is repeated 3x (different seed each repeat) to capture run-to-run
variance. Total: 4 datasets x 2 variants x 3 repeats = 24 YOLO trainings.

The stored data.yaml files point at a since-moved workspace path (they were
generated before these examples were archived under workspace/publication/...),
so this script writes a corrected copy (absolute "path" fixed) next to the
experiment output instead of mutating the original saved artifacts.

Usage (from repo root):
    python experiments/exp04_dataset_enricher_eval/run.py

Then summarize with:
    python experiments/exp04_dataset_enricher_eval/summarize.py
"""

import json
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp04_dataset_enricher_eval.config import (
    DATASETS, VARIANTS, N_REPEATS, MODEL_WEIGHTS, TRAIN_KWARGS, OUT_DIR,
    dataset_root, run_dir,
)
from src.funcs.dl_model_trainer_funcs.yolo_trainer import YOLOTrainer
from src.funcs.dl_model_trainer_funcs.yolo_evaluator import YOLOEvaluator


def prepare_fixed_yaml(variant_dir: Path, out_dir: Path) -> Path:
    """Copy data.yaml next to the experiment output with an up-to-date 'path'."""
    src_yaml = variant_dir / "data.yaml"
    with open(src_yaml, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = str(variant_dir.resolve())

    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(data, f)

    return out_yaml


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
    print(f"Experiment 04 — Dataset Enricher ablation — {total_runs} runs total")
    print(f"Datasets ({len(DATASETS)}): {[d['name'] for d in DATASETS]}")
    print(f"Variants: {list(VARIANTS.keys())}")
    print(f"Repeats per (dataset, variant): {N_REPEATS}")
    print(f"Train kwargs: {TRAIN_KWARGS}  model={MODEL_WEIGHTS}")

    manifest = []
    run_idx = 0

    for dataset in DATASETS[-2:-1]:
        root = dataset_root(dataset)

        for variant_key, variant_folder in list(VARIANTS.items())[1:]:
            variant_dir = root / variant_folder

            if not variant_dir.exists():
                print(f"[SKIP] missing variant dir: {variant_dir}")
                continue

            fixed_yaml_dir = OUT_DIR / "_fixed_yaml" / dataset["name"] / variant_key
            yaml_path = prepare_fixed_yaml(variant_dir, fixed_yaml_dir)

            for repeat in range(1, N_REPEATS + 1)[-1:]:
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
        print(f"  [{tag}] {e['dataset']:32s} [{e['variant']:20s}] repeat {e['repeat']}  {e['run_dir']}")

    print("\nNext: python experiments/exp04_dataset_enricher_eval/summarize.py")
