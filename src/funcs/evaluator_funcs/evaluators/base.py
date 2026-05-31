from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import json

from src.utils.logger import get_logger
from src.state.agent_state import EvalArtifact
from src.funcs.evaluator_funcs.utils.loaders import load_predictor

logger = get_logger(__name__)


class BaseEvaluator(ABC):

    metric_name = "unknown"

    # -------------------------------------------------
    # ENTRY POINT
    # -------------------------------------------------
    def evaluate(self, state):

        logger.info(f"Running {self.__class__.__name__}")

        predictor = load_predictor(state)

        if predictor is None:
            state["evaluation"] = {
                "metric_name": self.metric_name,
                "metric_value": 0.0,
                "status": "predictor_load_failed"
            }
            return state

        state["evaluation"] = {}
        state["evaluation_visualizations"] = {}

        for split in ["val", "test"]:

            result = self._evaluate_split(
                predictor=predictor,
                state=state,
                split=split
            )

            state["evaluation"][split] = result["metrics"]
            state["evaluation_visualizations"][split] = result["vis_paths"]

            if split == "val":
                self._store_eval_artifact(
                    state,
                    result["metrics"]["metric_value"],
                    result["vis_paths"]
                )

        logger.info(f"{self.__class__.__name__} finished")

        return state

    # -------------------------------------------------
    # CORE PIPELINE (no domain logic here)
    # -------------------------------------------------
    def _evaluate_split(self, predictor, state, split):

        image_dir = Path(state["split_paths"][split]["images"])
        label_dir = Path(state["split_paths"][split]["labels"])

        image_files = [p for p in image_dir.iterdir() if p.is_file()]

        all_predictions = []
        all_ground_truths = {}

        # 1. COLLECT DATA
        for img_path in image_files:

            preds = self._safe_predict(predictor, img_path)
            if preds is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gts = self.load_gt(label_dir, img_path.stem, img.shape[:2])

            all_ground_truths[img_path.name] = gts

            for p in preds:
                all_predictions.append({
                    "image": img_path.name,
                    **self.normalize_prediction(p)
                })

        # 2. SORT BY SCORE
        all_predictions = sorted(all_predictions, key=lambda x: x.get("score", 1.0), reverse=True)

        # 3. MATCHING (delegated to matcher module)
        tp_all, fp_all, fn_all = self.match(predictions=all_predictions, ground_truths=all_ground_truths)

        # 4. METRICS (delegated)
        metrics = self.compute_metric(tp_all=tp_all, fp_all=fp_all, fn_all=fn_all, all_gts=all_ground_truths)

        # 5. VISUALIZATION (delegated OUTSIDE evaluator logic)
        vis_paths = self._create_visualizations(image_files=image_files, tp_all=tp_all, fp_all=fp_all,
                                                fn_all=fn_all, state=state, split=split)

        # 6. SAVE METRICS
        self._save_metrics(state, split, metrics)

        return {
            "metrics": metrics,
            "vis_paths": vis_paths
        }

    # -------------------------------------------------
    # ABSTRACT (domain-specific only)
    # -------------------------------------------------
    @abstractmethod
    def load_gt(self, label_dir, img_stem, img_shape):
        pass

    @abstractmethod
    def match(self, predictions, ground_truths):
        pass

    @abstractmethod
    def compute_metric(self, tp_all, fp_all, fn_all, all_gts):
        pass

    @abstractmethod
    def visualize_single(self, image_path, tp_all, fp_all, fn_all):
        pass

    # -------------------------------------------------
    # NORMALIZATION (shared across all evaluators)
    # -------------------------------------------------
    def normalize_prediction(self, pred: dict):
        return {
            "bbox": pred.get("bbox"),
            "point": pred.get("point"),
            "line": pred.get("line"),
            "score": pred.get("confidence", 1.0),
            "label": pred.get("label", None)
        }

    # -------------------------------------------------
    # VISUALIZATION PIPELINE (uses external module logic)
    # -------------------------------------------------
    def _create_visualizations(self, image_files, tp_all, fp_all, fn_all, state, split):

        vis_dir = Path(
            f"workspace/"
            f"stage_{state.get('stage_id', 0)}"
            f"_step_{state.get('step_id', 0)}"
            f"/evaluation/visualizations/{split}"
        )

        vis_dir.mkdir(parents=True, exist_ok=True)

        vis_paths = []

        for img_path in image_files:

            vis = self.visualize_single(img_path, tp_all, fp_all, fn_all)

            if vis is None:
                continue

            out_path = vis_dir / f"{img_path.stem}_eval.jpg"

            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            vis_paths.append(str(out_path))

        return vis_paths

    # -------------------------------------------------
    # METRICS SAVE
    # -------------------------------------------------
    def _save_metrics(self, state, split, metrics):

        metrics_dir = Path(
            f"workspace/"
            f"stage_{state.get('stage_id', 0)}"
            f"_step_{state.get('step_id', 0)}"
            f"/evaluation/metrics/{split}"
        )

        metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(metrics_dir / f"{split}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------
    # ARTIFACT STORAGE
    # -------------------------------------------------
    def _store_eval_artifact(self, state, value, vis_paths):

        step_key = (
            f"stage_{state.get('stage_id', 0)}"
            f"_step_{state.get('step_id', 0)}"
        )

        state["eval_artifacts"].append(
            EvalArtifact(
                step_key=step_key,
                value=value,
                img_paths=vis_paths
            )
        )

    # -------------------------------------------------
    # SAFE PREDICT
    # -------------------------------------------------
    def _safe_predict(self, predictor, img_path):
        try:
            return predictor.predict(str(img_path))
        except Exception as e:
            logger.exception(f"Prediction failed: {img_path} -> {e}")
            return None
