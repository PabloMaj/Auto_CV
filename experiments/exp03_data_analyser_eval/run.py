"""
Experiment 03 — DataAnalyserAgent output-format classification eval

Evaluates `determine_desired_output` (the core decision function used by
DataAnalyserAgent) on 20 hand-written user_prompt examples for each of:
"points", "line_segments", "bounding_boxes".

For every example the model must pick the correct desired_output from the
full allowed list (scalar, points, line_segments, bounding_boxes, polygons,
segmentation_masks). Accuracy per category + a confusion matrix are reported.

Usage (from repo root):
    python experiments/exp03_data_analyser_eval/run.py
    python experiments/exp03_data_analyser_eval/run.py --backend ollama --model qwen2.5vl:7b
    python experiments/exp03_data_analyser_eval/run.py --backend sonnet --model claude-haiku-4-5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp03_data_analyser_eval.examples import EXAMPLES
from src.funcs.data_analyser_funcs.data_analyser_funcs import determine_desired_output
from src.inference.factory import InferenceFactory

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = _REPO_ROOT / "workspace" / "exp03_data_analyser_eval"

# Must match the allowed_outputs list used in DataAnalyserAgent.
OUTPUT_FORMS_ALLOWED = ["scalar", "points", "line_segments", "bounding_boxes", "polygons", "segmentation_masks"]

CATEGORIES = ["points", "line_segments", "bounding_boxes"]

DEFAULT_MODEL = {
    "ollama": "qwen2.5vl:7b",
    "sonnet": "claude-haiku-4-5",
}


def build_inference(backend: str, model: str):
    if backend == "ollama":
        return InferenceFactory.create(backend="ollama", model=model)
    return InferenceFactory.create(backend="sonnet", model=model, temperature=0.0, max_tokens=16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "sonnet"], default="ollama")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL[args.backend]
    inference = build_inference(args.backend, model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    confusion = {expected: {} for expected in CATEGORIES}
    records = []

    print(f"Experiment 03 — DataAnalyserAgent classification eval  [{args.backend}/{model}]")
    print(f"Categories: {CATEGORIES}  (20 examples each, {len(CATEGORIES) * 20} total)\n")

    correct = 0
    total = 0

    for expected in CATEGORIES:
        print(f"{'=' * 60}\nCATEGORY: {expected}\n{'=' * 60}")
        cat_correct = 0

        for i, prompt in enumerate(EXAMPLES[expected]):
            predicted = determine_desired_output(inference, prompt, OUTPUT_FORMS_ALLOWED)
            is_correct = predicted == expected
            cat_correct += int(is_correct)
            correct += int(is_correct)
            total += 1

            confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
            records.append({
                "idx": i,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "prompt": prompt,
            })

            tag = "OK " if is_correct else "ERR"
            print(f"  [{tag}] [{i + 1:02d}/20] predicted={predicted:16s} | {prompt[:70]}")

        print(f"  → category accuracy: {cat_correct}/20 ({cat_correct / 20:.1%})\n")

    overall_acc = correct / total

    # --- confusion matrix printout ---
    all_labels = sorted({label for row in confusion.values() for label in row} | set(CATEGORIES))
    print(f"{'=' * 60}\nCONFUSION MATRIX (rows=expected, cols=predicted)\n{'=' * 60}")
    header = " " * 18 + "".join(f"{lbl:>16s}" for lbl in all_labels)
    print(header)
    for expected in CATEGORIES:
        row = "".join(f"{confusion[expected].get(lbl, 0):>16d}" for lbl in all_labels)
        print(f"{expected:18s}{row}")

    print(f"\nOVERALL ACCURACY: {correct}/{total} ({overall_acc:.1%})")

    # --- save results ---
    results = {
        "timestamp": timestamp,
        "backend": args.backend,
        "model": model,
        "categories": CATEGORIES,
        "n_per_category": 20,
        "overall_accuracy": round(overall_acc, 4),
        "category_accuracy": {
            expected: round(sum(1 for r in records if r["expected"] == expected and r["correct"]) / 20, 4)
            for expected in CATEGORIES
        },
        "confusion_matrix": confusion,
        "records": records,
    }

    results_path = OUT_DIR / f"results_{args.backend}_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved -> {results_path}")


if __name__ == "__main__":
    main()
