"""
Summary script for Experiment 04 — Dataset Enricher ablation.

Reads test_metrics.json directly from each run directory (not manifest.json —
manifest.json is overwritten on every run.py invocation, so it only reflects
the most recent invocation and is unreliable once runs happen across multiple
sessions or get re-run with different settings). Reports, for variant A
(tiled_dataset, no enrichment) vs variant B (final_dataset, pseudo-label
enrichment):

  1. Per-dataset metrics: mean +/- std over the 3 repeats.
  2. Summary averaged over datasets: mean +/- std of the 4 per-dataset means.

Usage (from repo root, after run.py has produced at least some results):
    python experiments/exp04_dataset_enricher_eval/summarize.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp04_dataset_enricher_eval.config import DATASETS, VARIANTS, N_REPEATS, OUT_DIR, run_dir


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def collect_runs():
    """Scan each run directory on disk for test_metrics.json (written by
    YOLOEvaluator.test()). Returns (successful, missing) where each entry is
    {"dataset", "variant", "repeat", "run_dir", "test_metrics"} /
    {"dataset", "variant", "repeat", "run_dir"} respectively."""
    successful = []
    missing = []

    for dataset in DATASETS:
        ds_name = dataset["name"]
        for variant_key in VARIANTS:
            for repeat in range(1, N_REPEATS + 1):
                out_dir = run_dir(ds_name, variant_key, repeat)
                metrics_path = out_dir / "test_metrics.json"

                if not metrics_path.exists():
                    missing.append({
                        "dataset": ds_name, "variant": variant_key,
                        "repeat": repeat, "run_dir": str(out_dir),
                    })
                    continue

                with open(metrics_path, "r", encoding="utf-8") as f:
                    test_metrics = json.load(f)

                successful.append({
                    "dataset": ds_name, "variant": variant_key,
                    "repeat": repeat, "run_dir": str(out_dir),
                    "test_metrics": test_metrics,
                })

    return successful, missing


def main():
    successful, missing = collect_runs()

    if not successful:
        print(f"No test_metrics.json found under {OUT_DIR}. Run run.py first.")
        return

    if missing:
        print(f"[WARNING] {len(missing)} run(s) have no test_metrics.json yet and are excluded from the summary:")
        for e in missing:
            print(f"  - {e['dataset']} [{e['variant']}] repeat {e['repeat']}: {e['run_dir']}")
        print()

    metric_keys = sorted({
        k for e in successful for k, v in e["test_metrics"].items()
        if isinstance(v, (int, float))
    })

    dataset_names = [d["name"] for d in DATASETS]
    variant_keys = list(VARIANTS.keys())

    # dataset -> variant -> {"n_runs": int, "metrics": {key: {"mean": .., "std": ..}}}
    per_dataset_summary = {}
    for ds_name in dataset_names:
        per_dataset_summary[ds_name] = {}
        for variant in variant_keys:
            rows = [e for e in successful if e["dataset"] == ds_name and e["variant"] == variant]
            per_dataset_summary[ds_name][variant] = {
                "n_runs": len(rows),
                "metrics": {
                    key: dict(zip(("mean", "std"), mean_std([r["test_metrics"].get(key) for r in rows])))
                    for key in metric_keys
                },
            }

    # --- print per-dataset tables ---
    print("=" * 100)
    print("PER-DATASET SUMMARY (mean +/- std over repeats)")
    print("=" * 100)
    for ds_name in dataset_names:
        print(f"\n--- {ds_name} ---")
        print(f"{'metric':30s}" + "".join(f"{v:>28s}" for v in variant_keys))
        for key in metric_keys:
            row = f"{key:30s}"
            for variant in variant_keys:
                stats = per_dataset_summary[ds_name][variant]["metrics"][key]
                cell = "n/a" if stats["mean"] is None else f"{stats['mean']:.4f} +/- {stats['std']:.4f}"
                row += f"{cell:>28s}"
            print(row)
        n_runs_line = "  ".join(f"{v}: n={per_dataset_summary[ds_name][v]['n_runs']}" for v in variant_keys)
        print(f"  ({n_runs_line})")

    # --- summary averaged over datasets (equal weight per dataset) ---
    overall_summary = {}
    for variant in variant_keys:
        overall_summary[variant] = {}
        for key in metric_keys:
            per_ds_means = [
                per_dataset_summary[ds_name][variant]["metrics"][key]["mean"]
                for ds_name in dataset_names
                if per_dataset_summary[ds_name][variant]["metrics"][key]["mean"] is not None
            ]
            mean, std = mean_std(per_ds_means)
            overall_summary[variant][key] = {"mean": mean, "std": std, "n_datasets": len(per_ds_means)}

    print("\n" + "=" * 100)
    print("SUMMARY AVERAGED OVER DATASETS (mean +/- std across the per-dataset means)")
    print("=" * 100)
    print(f"{'metric':30s}" + "".join(f"{v:>28s}" for v in variant_keys))
    for key in metric_keys:
        row = f"{key:30s}"
        for variant in variant_keys:
            stats = overall_summary[variant][key]
            cell = "n/a" if stats["mean"] is None else f"{stats['mean']:.4f} +/- {stats['std']:.4f}"
            row += f"{cell:>28s}"
        print(row)

    # --- save ---
    out = {
        "per_dataset": per_dataset_summary,
        "averaged_over_datasets": overall_summary,
        "n_missing_runs": len(missing),
        "missing_runs": missing,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
