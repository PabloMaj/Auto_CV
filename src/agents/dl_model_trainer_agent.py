from pathlib import Path

from src.funcs.dl_model_trainer_funcs.yolo_pipeline import YOLOPipeline
from src.utils.cuda import cuda_cleanup
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DLModelTrainerAgent:

    def __init__(self, settings=None):
        self.settings = settings

    def run(self, state):
        logger.info("Running DLModelTrainerAgent")

        dl_path_obj = Path(state.get("dl_dataset_path"))
        dl_dataset_name = dl_path_obj.name
        dl_dataset_group = dl_path_obj.parent.name

        repo_root = Path(__file__).resolve().parents[2]
        exp_id = state.get("exp_id", "default")
        exp_workspace = repo_root / "workspace" / exp_id

        artifacts_path = exp_workspace / "yolo_train_artifacts" / dl_dataset_group / f"{dl_dataset_name}_roi640"
        processed_dataset = exp_workspace / "processed_datasets" / dl_dataset_group / f"{dl_dataset_name}_roi640"

        yolo_weights = self.settings.yolo_model_weights if self.settings else "yolo11m.pt"

        pipeline = YOLOPipeline(
            raw_dataset=dl_path_obj,
            processed_dataset=processed_dataset,
            yolo_train_artifacts_save_path=artifacts_path,
            class_names=["object"],
            model_weights=yolo_weights,
            pseudo_images_dir=exp_workspace / "dataset_enrichment_pseudo" / dl_dataset_group / dl_dataset_name / "tiled_unlabeled" / "images",
            pseudo_labels_dir=exp_workspace / "dataset_enrichment_pseudo" / dl_dataset_group / dl_dataset_name / "pseudo_labels",
        )
        pipeline.run()
        del pipeline
        cuda_cleanup("dl_model_trainer")

        state["yolo_model_path"] = str(artifacts_path / "weights" / "best.pt")

        return state
