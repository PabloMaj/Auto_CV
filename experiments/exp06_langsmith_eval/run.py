"""
Experiment 06 — LangSmith tracked experiment over the full agent graph.

Runs the whole agent_cv graph (build_graph) once per example of the
LangSmith dataset "agent_cv_dataset" (exported from
workspace/langsmith_exports/dataset_examples.jsonl, 3 examples: one per
variant — bboxes / midpoints / lines) and records it as a LangSmith
experiment. Each run is fully traced (LangGraph node-by-node) because
setup_langsmith_tracing() enables tracing before the graph is built.

By default, scoring is done by the 5 LOCAL_EVALUATORS below, run in this
process by langsmith.evaluate() itself — no UI setup needed. evaluate()
uploads each evaluator's returned {"key", "score"} as feedback on the
matching experiment run, so results still show up as columns in the
LangSmith UI experiment table. Pass --no-local-evaluators to skip this and
rely solely on evaluators bound to the dataset in the LangSmith UI instead
(see workspace/langsmith_exports/evaluators/ for that UI-paste format —
useful if you want the same evaluator to also apply to experiments created
directly from the UI, not just from this script).

By default this trims the graph to 2 stages x 2 improvement steps
(--max-novel-solutions / --max-improvement-steps, production defaults are
2 / 5 in src/config/settings.py) to keep experiment runs short/cheap. Bump
them back up (or to whatever) for a closer-to-production run.

Prerequisites:
  - LANGSMITH_API_KEY set in the environment.
  - SONNET_API_KEY set in the environment (programmer_llm / improvement_llm /
    judge_llm all use the "sonnet" backend -> src/inference/sonnet_inference.py).
  - Ollama running locally with the qwen2.5vl:7b model pulled (DataAnalyserAgent
    always uses it, regardless of settings.vision_llm).
  - The "agent_cv_dataset" dataset must already exist in your LangSmith project
    (per the user: already created from dataset_examples.jsonl).

Usage (from repo root):
    python experiments/exp06_langsmith_eval/run.py
    python experiments/exp06_langsmith_eval/run.py --max-concurrency 1 --experiment-prefix agent_cv-smoke
    python experiments/exp06_langsmith_eval/run.py --max-improvement-steps 5 --max-novel-solutions 2  # closer to production settings.py
    python experiments/exp06_langsmith_eval/run.py --no-local-evaluators  # score only via evaluators bound in the LangSmith UI
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.langsmith_integration import setup_langsmith_tracing  # noqa: E402

setup_langsmith_tracing()

from langsmith import evaluate  # noqa: E402

from src.graph.workflow import build_graph  # noqa: E402
from src.state.agent_state import AgentState  # noqa: E402
from src.config.settings import SystemSettings  # noqa: E402
from src.utils.state_utils import state_to_json  # noqa: E402

DATASET_NAME = "agent_cv_dataset"


# ---------------------------------------------------------------------------
# Target: one full graph run per dataset example
# ---------------------------------------------------------------------------

def build_target(settings: SystemSettings):

    def target(inputs: dict) -> dict:
        # dataset image_path -> dataset root dir, e.g.
        # data/data_structured/crop_line_uav/sunflower_1_auzeville_2019_1_bboxes/images/train/image_001.jpg
        # -> data/data_structured/crop_line_uav/sunflower_1_auzeville_2019_1_bboxes
        dl_dataset_path = str(Path(inputs["image_path"]).parents[2])

        initial_state = AgentState(
            user_prompt=inputs["user_prompt"],
            dl_dataset_path=dl_dataset_path,
            exp_id=f"langsmith_{inputs.get('variant', 'na')}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        )

        graph = build_graph(settings)
        final_state = state_to_json(graph.invoke(initial_state.model_dump()))

        exp_dir = Path("workspace") / initial_state.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "final_state.json", "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2, ensure_ascii=False)

        # Only the fields the 5 evaluators need (keeps the LangSmith run
        # payload small and matches src/config/langsmith_integration.py docs).
        return {
            "desired_output": final_state.get("desired_output"),
            "generated_code": final_state.get("generated_code"),
            "improvement_suggestions": final_state.get("improvement_suggestions"),
            "train_samples": final_state.get("train_samples"),
            "val_samples": final_state.get("val_samples"),
            "test_samples": final_state.get("test_samples"),
            "unlabelled_samples": final_state.get("unlabelled_samples"),
            "train_objects": final_state.get("train_objects"),
            "val_objects": final_state.get("val_objects"),
            "test_objects": final_state.get("test_objects"),
            "unlabelled_objects": final_state.get("unlabelled_objects"),
            "evaluation": final_state.get("evaluation"),
            "exp_id": initial_state.exp_id,
        }

    return target


# ---------------------------------------------------------------------------
# The 5 evaluators, run locally by evaluate() and tracked to the LangSmith UI
# as feedback on each experiment run (default; disable with --no-local-evaluators).
# Same rules as workspace/langsmith_exports/evaluators/, but using the
# (inputs, outputs, reference_outputs) signature the SDK's evaluate() expects
# instead of perform_eval(run, example), which only the UI code editor requires.
# ---------------------------------------------------------------------------

def task_type_correct(inputs, outputs, reference_outputs):
    mapping = {"bboxes": "bounding_boxes", "midpoints": "points", "line_segments": "line_segments"}
    expected = mapping.get((reference_outputs or {}).get("expected_task_type"))
    actual = (outputs or {}).get("desired_output")
    return {"key": "task_type_correct", "score": int(actual == expected)}


def generated_code_length_ok(inputs, outputs, reference_outputs):
    length = len((outputs or {}).get("generated_code") or "")
    return {"key": "generated_code_length_ok", "score": int(length >= 100)}


def improvement_suggestions_length_ok(inputs, outputs, reference_outputs):
    suggestions = (outputs or {}).get("improvement_suggestions") or []
    length = len("\n".join(str(s) for s in suggestions))
    return {"key": "improvement_suggestions_length_ok", "score": int(length >= 100)}


def preprocessing_has_samples_and_objects(inputs, outputs, reference_outputs):
    outputs = outputs or {}
    sample_keys = ["train_samples", "val_samples", "test_samples", "unlabelled_samples"]
    object_keys = ["train_objects", "val_objects", "test_objects", "unlabelled_objects"]
    total_samples = sum(outputs.get(k, 0) or 0 for k in sample_keys)
    total_objects = sum(outputs.get(k, 0) or 0 for k in object_keys)
    return {"key": "preprocessing_has_samples_and_objects", "score": int(total_samples > 0 and total_objects > 0)}


def val_metric_computed(inputs, outputs, reference_outputs):
    evaluation = (outputs or {}).get("evaluation") or {}
    val_block = evaluation.get("val")
    metric_value = val_block.get("metric_value") if isinstance(val_block, dict) else evaluation.get("metric_value")
    is_number = isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool)
    return {"key": "val_metric_computed", "score": int(is_number)}


LOCAL_EVALUATORS = [
    task_type_correct,
    generated_code_length_ok,
    improvement_suggestions_length_ok,
    preprocessing_has_samples_and_objects,
    val_metric_computed,
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--experiment-prefix", default="agent_cv-full_graph")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--no-local-evaluators", action="store_true",
        help="Skip computing the 5 evaluators client-side; rely solely on evaluators bound to the dataset in the LangSmith UI instead.",
    )
    parser.add_argument(
        "--max-improvement-steps", type=int, default=2,
        help="Steps per stage before moving to the next stage (settings.max_improvement_steps, default 5 in production).",
    )
    parser.add_argument(
        "--max-novel-solutions", type=int, default=2,
        help="Number of stages run before stopping (settings.max_novel_solutions; route_after_improvement stops once stage_id exceeds this).",
    )
    args = parser.parse_args()

    settings = SystemSettings(
        max_improvement_steps=args.max_improvement_steps,
        max_novel_solutions=args.max_novel_solutions,
    )

    results = evaluate(
        build_target(settings),
        data=args.dataset,
        evaluators=None if args.no_local_evaluators else LOCAL_EVALUATORS,
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        metadata={"settings": settings.model_dump()},
    )

    print(f"\nExperiment submitted: {results.experiment_name}")
    print(f"View at: {results.url}")


if __name__ == "__main__":
    main()
