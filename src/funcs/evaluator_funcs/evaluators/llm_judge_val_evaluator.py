import json
from pathlib import Path

import cv2
import numpy as np

from src.funcs.evaluator_funcs.utils.loaders import load_predictor
from src.prompts.evaluator_prompts.llm_judge_prompt import (
    LLM_JUDGE_SYSTEM_PROMPT,
    LLM_JUDGE_USER_TEMPLATE,
)
from src.state.agent_state import EvalArtifact
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PRED_COLOR = (255, 200, 0)   # cyan (BGR) — all predictions drawn in this colour
_MAX_JUDGE_IMAGES = 2


class LLMJudgeValEvaluator:
    """Evaluates the val split without ground-truth labels using an LLM as judge."""

    def __init__(self, llm):
        self.llm = llm

    def evaluate_val(self, state) -> dict:
        predictor = load_predictor(state)
        if predictor is None:
            logger.warning("LLMJudgeValEvaluator: predictor load failed — skipping val")
            return state

        split_paths = state.get("split_paths", {})
        image_dir = Path(split_paths.get("val", {}).get("images", ""))
        user_prompt = state.get("user_prompt", "")
        desired_out = state.get("desired_output", "unknown")
        stage_id = state.get("stage_id", 0)
        step_id = state.get("step_id", 0)

        if not image_dir.exists():
            logger.warning(f"Val image dir not found: {image_dir}")
            return state

        repo_root = Path(__file__).resolve().parents[4]
        exp_workspace = repo_root / "workspace" / state.get("exp_id", "default")
        self.llm.log_dir = exp_workspace / "llm_logs" / "llm_judge" / f"stage_{stage_id}_step_{step_id}"
        vis_dir = exp_workspace / f"stage_{stage_id}_step_{step_id}/evaluation/visualizations/val"
        metrics_dir = exp_workspace / f"stage_{stage_id}_step_{step_id}/evaluation/metrics/val"
        vis_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])[:_MAX_JUDGE_IMAGES]

        scores = []
        vis_paths = []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            try:
                preds = predictor.predict(str(img_path))
            except Exception as exc:
                logger.warning(f"Prediction failed for {img_path}: {exc}")
                continue

            vis = _draw_predictions(img.copy(), preds, desired_out)

            out = vis_dir / f"{img_path.stem}_eval.jpg"
            cv2.imwrite(str(out), vis)
            vis_paths.append(str(out))

            score = self._query_judge(vis, img_path, user_prompt, desired_out)
            scores.append(score)

        mean_score = float(np.mean(scores)) if scores else 0.0

        metrics = {
            "metric_name": "LLM_JUDGE_SCORE",
            "metric_value": mean_score,
            "num_evaluated": len(scores),
            "status": "success",
        }

        if not isinstance(state.get("evaluation"), dict):
            state["evaluation"] = {}
        if not isinstance(state.get("evaluation_visualizations"), dict):
            state["evaluation_visualizations"] = {}

        state["evaluation"]["val"] = metrics
        state["evaluation_visualizations"]["val"] = vis_paths

        with open(metrics_dir / "val_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        step_key = f"stage_{stage_id}_step_{step_id}"
        state["eval_artifacts"].append(
            EvalArtifact(step_key=step_key, value=mean_score, img_paths=vis_paths)
        )

        logger.info(f"LLM judge val score: {mean_score:.4f} ({len(scores)} images)")
        return state

    def _query_judge(self, image: np.ndarray, img_path: Path,
                     user_prompt: str, output_type: str) -> float:
        tmp = Path(img_path).parent / f"_judge_tmp_{Path(img_path).stem}.jpg"
        cv2.imwrite(str(tmp), image)
        try:
            prompt = LLM_JUDGE_USER_TEMPLATE.format(
                user_prompt=user_prompt, output_type=output_type
            )
            messages = self.llm.build_messages(
                prompt=prompt,
                image_paths=[str(tmp)],
                system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
            )
            raw = self.llm.infer(messages=messages)
            data = _parse_json(raw)
            return float(data.get("metric_value", 0.0))
        except Exception as exc:
            logger.warning(f"LLM judge query failed: {exc}")
            return 0.0
        finally:
            if tmp.exists():
                tmp.unlink()


# ---------------------------------------------------------------------------
# Drawing helper — single colour, no labels
# ---------------------------------------------------------------------------

def _draw_predictions(img: np.ndarray, preds: list, output_type: str) -> np.ndarray:
    for pred in preds:
        if output_type == "line_segments":
            seg = pred.get("line") or pred.get("line_segment")
            if seg and len(seg) == 4:
                x1, y1, x2, y2 = [int(v) for v in seg]
                cv2.line(img, (x1, y1), (x2, y2), _PRED_COLOR, 2)

        elif output_type == "bounding_boxes":
            bbox = pred.get("bbox") or pred.get("bounding_box")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), _PRED_COLOR, 2)

        elif output_type in ("points", "midpoints"):
            pt = pred.get("point") or pred.get("midpoint")
            if pt and len(pt) >= 2:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 7, _PRED_COLOR, -1)

        else:
            for key in ("line", "bbox", "point", "midpoint"):
                val = pred.get(key)
                if val:
                    if len(val) == 4:
                        cv2.line(img, (int(val[0]), int(val[1])),
                                 (int(val[2]), int(val[3])), _PRED_COLOR, 2)
                    elif len(val) == 2:
                        cv2.circle(img, (int(val[0]), int(val[1])), 7, _PRED_COLOR, -1)
                    break

    return img


def _parse_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"metric_value": 0.0}
