"""
Experiment 02 — LLM-as-Judge Prompt Calibration

For each variant (bboxes / midpoints / lines):
  1. Generate 20 test cases with ground-truth F1 in [0.40, 1.0].
  2. For each of 5 judge prompts, query Sonnet and record the predicted score.
  3. Plot predicted score vs actual F1 (scatter, 20 points per prompt).
  4. Compute R² for each prompt.
  5. Select the best prompt per variant.

Usage (from repo root):
    python experiments/exp02_llm_judge_calibration/run.py
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp02_llm_judge_calibration.case_gen import generate_test_cases
from experiments.exp02_llm_judge_calibration.prompts_typed import PROMPTS_TYPED as PROMPTS
from src.inference.sonnet_inference import (
    SonnetInference,
    _tracker,
    reset_cost_tracker,
    save_cost_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = _REPO_ROOT / "data/data_structured/crop_line_uav"
OUT_DIR = _REPO_ROOT / "workspace/exp02_llm_judge_calibration"
N_CASES = 20
SEED = 42
MODEL = "claude-opus-4-8"

VARIANT_USER_PROMPT = {
    "bboxes": (
        "Detect individual crop plants in UAV RGB images. "
        "Return bounding boxes around each plant."
    ),
    "midpoints": (
        "Detect individual crop plants in UAV RGB images. "
        "Return the center point (midpoint) of each plant."
    ),
    "lines": (
        "Detect crop row lines in UAV RGB images. "
        "Return one line segment per visible crop row."
    ),
}

VARIANT_OUTPUT_TYPE = {
    "bboxes": "bounding_boxes",
    "midpoints": "midpoints",
    "lines": "line_segments",
}


# ---------------------------------------------------------------------------
# Cost reporting
# ---------------------------------------------------------------------------

def _cost_summary() -> str:
    inp = _tracker["input_tokens"]
    out = _tracker["output_tokens"]
    cost = (inp * 5.0 + out * 25.0) / 1_000_000
    return (f"calls={_tracker['calls']}  in={inp}  out={out}  "
            f"cost=${cost:.4f}")


# ---------------------------------------------------------------------------
# Judge query
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return {}


def query_judge(
    sonnet: SonnetInference,
    vis_bgr: np.ndarray,
    prompt_cfg: dict,
    variant: str,
    tmp_dir: Path,
    case_idx: int,
    prompt_name: str,
    n_pred: int = 0,
) -> float:
    tmp_path = tmp_dir / f"{prompt_name}_{case_idx}.jpg"
    cv2.imwrite(str(tmp_path), vis_bgr)

    try:
        user_text = prompt_cfg["user_template"].format(
            user_prompt=VARIANT_USER_PROMPT[variant],
            output_type=VARIANT_OUTPUT_TYPE[variant],
            n_pred=n_pred,
        )
        messages = sonnet.build_messages(
            prompt=user_text,
            image_paths=[str(tmp_path)],
            system_prompt=prompt_cfg["system"],
        )
        raw = sonnet.infer(messages=messages)
        data = _parse_json(raw)
        return float(data.get("metric_value", 0.0))
    except Exception as exc:
        print(f"    [ERROR] judge query failed: {exc}")
        return 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# R²
# ---------------------------------------------------------------------------

def compute_r2(y_true: list[float], y_pred: list[float]) -> float:
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < 1e-9:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_prompt(
    variant: str, prompt_name: str, actual_f1s: list, scores: list, r2: float, out_dir: Path
):
    """Save individual scatter plot for one prompt right after evaluation."""
    label = PROMPTS[prompt_name]["label"]
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(actual_f1s, scores, alpha=0.75, s=60, zorder=3)
    ax.plot([0.3, 1.05], [0.3, 1.05], "r--", alpha=0.4, label="ideal")
    ax.set_xlabel("Actual F1 (ground truth)")
    ax.set_ylabel("LLM Judge Score")
    ax.set_title(f"{variant} — {label}\nR² = {r2:.4f}", fontsize=10)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_path = out_dir / f"scatter_{variant}_{prompt_name}.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter saved → {plot_path}")


def plot_variant(variant: str, test_cases, prompt_results: dict, out_dir: Path):
    """Save combined figure with all prompts side by side."""
    if not prompt_results:
        return

    n_prompts = len(prompt_results)
    fig, axes = plt.subplots(1, n_prompts, figsize=(5 * n_prompts, 5), sharey=True)
    if n_prompts == 1:
        axes = [axes]

    actual_f1s = [tc.actual_f1 for tc in test_cases]

    for ax, (prompt_name, data) in zip(axes, prompt_results.items()):
        scores = data["scores"]
        r2 = data["r2"]
        label = PROMPTS[prompt_name]["label"]

        ax.scatter(actual_f1s, scores, alpha=0.75, s=60, zorder=3)
        ax.plot([0.3, 1.05], [0.3, 1.05], "r--", alpha=0.4, label="ideal")
        ax.set_xlabel("Actual F1 (ground truth)")
        ax.set_title(f"{label}\nR² = {r2:.4f}", fontsize=9)
        ax.set_xlim(0.3, 1.05)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("LLM Judge Score")
    fig.suptitle(f"Exp 02 — LLM Judge Calibration [{variant}]", fontsize=12)
    plt.tight_layout()

    plot_path = out_dir / f"calibration_{variant}.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "_tmp_vis"
    tmp_dir.mkdir(exist_ok=True)

    reset_cost_tracker()

    sonnet = SonnetInference(
        model=MODEL,
        temperature=None,
        max_tokens=512,
        max_retries=3,
    )

    all_results: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant in ["bboxes", "midpoints", "lines"]:
        print(f"\n{'=' * 60}")
        print(f"VARIANT: {variant}")
        print(f"{'=' * 60}")

        # --- generate test cases ---
        try:
            test_cases = generate_test_cases(
                variant=variant,
                data_root=DATA_ROOT,
                n_cases=N_CASES,
                seed=SEED,
            )
        except Exception as exc:
            print(f"  [ERROR] case generation failed: {exc}")
            traceback.print_exc()
            continue

        if not test_cases:
            print("  [SKIP] no test cases generated")
            continue

        # --- save visualizations and metadata ---
        cases_dir = OUT_DIR / "cases" / variant
        cases_dir.mkdir(parents=True, exist_ok=True)

        cases_meta = []
        for i, tc in enumerate(test_cases):
            vis_name = f"{i:02d}_f1_{tc.actual_f1:.3f}_{tc.dataset.replace('_' + variant, '')}.jpg"
            vis_path = cases_dir / vis_name
            cv2.imwrite(str(vis_path), tc.vis_bgr)

            cases_meta.append({
                "idx": i,
                "vis_image": str(vis_path),
                "source_image": str(tc.image_path),
                "dataset": tc.dataset,
                "actual_f1": round(tc.actual_f1, 5),
                "n_gt": len(tc.gt_annotations),
                "n_pred": len(tc.predictions),
                "gt_annotations": tc.gt_annotations,
                "predictions": tc.predictions,
            })

        meta_path = OUT_DIR / f"test_cases_{variant}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cases_meta, f, indent=2)
        print(f"  Saved {len(test_cases)} visualizations → {cases_dir}")

        variant_results: dict = {}

        # --- evaluate each prompt (skip prompts tagged for a different variant) ---
        variant_prompts = {
            name: cfg for name, cfg in PROMPTS.items()
            if cfg.get("variant_only", variant) == variant
        }
        for prompt_name, prompt_cfg in variant_prompts.items():
            print(f"\n  Prompt: {prompt_name!r}  ({prompt_cfg['label']})")
            scores = []

            for i, tc in enumerate(test_cases):
                score = query_judge(
                    sonnet=sonnet,
                    vis_bgr=tc.vis_bgr,
                    prompt_cfg=prompt_cfg,
                    variant=variant,
                    tmp_dir=tmp_dir,
                    case_idx=i,
                    prompt_name=prompt_name,
                    n_pred=len(tc.predictions),
                )
                scores.append(score)

                print(
                    f"    [{i + 1:02d}/{len(test_cases)}]  "
                    f"actual_F1={tc.actual_f1:.3f}  judge={score:.3f}  "
                    f"dataset={tc.dataset.replace('_' + variant, '')[:30]}"
                )
                print(f"         cost so far → {_cost_summary()}")

            actual_f1s = [tc.actual_f1 for tc in test_cases]
            r2 = compute_r2(actual_f1s, scores)
            print(f"  ── R² = {r2:.4f}")

            variant_results[prompt_name] = {
                "label": prompt_cfg["label"],
                "scores": [round(s, 4) for s in scores],
                "r2": round(r2, 4),
            }

            # save individual scatter immediately after each prompt finishes
            plot_single_prompt(variant, prompt_name, actual_f1s, scores, r2, OUT_DIR)

        all_results[variant] = {
            "test_cases": cases_meta,
            "prompts": variant_results,
        }

        # --- plot ---
        plot_variant(variant, test_cases, variant_results, OUT_DIR)

    # --- save results ---
    results_path = OUT_DIR / f"results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    save_cost_report(OUT_DIR / "cost_report.json")

    # --- summary ---
    print(f"\n{'=' * 60}")
    print("BEST PROMPT PER VARIANT (highest R²):")
    summary_rows = []
    for variant, res in all_results.items():
        prompts_r2 = {p: d["r2"] for p, d in res["prompts"].items()}
        best_name = max(prompts_r2, key=prompts_r2.get)
        best_r2 = prompts_r2[best_name]
        label = PROMPTS[best_name]["label"]
        print(f"  {variant:10s}  →  {best_name:20s}  R²={best_r2:.4f}  ({label})")
        summary_rows.append({
            "variant": variant,
            "best_prompt": best_name,
            "best_r2": best_r2,
            "all_r2": prompts_r2,
        })

    summary = {
        "timestamp": timestamp,
        "model": MODEL,
        "n_cases_per_variant": N_CASES,
        "data_root": str(DATA_ROOT),
        "best_prompts": summary_rows,
        "total_cost": _cost_summary(),
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults → {results_path}")
    print(f"Summary → {summary_path}")
    print(f"\nFinal cost: {_cost_summary()}")

    # cleanup tmp
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
