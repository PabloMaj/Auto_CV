from pathlib import Path

from src.utils.logger import get_logger
from src.funcs.dl_model_trainer_funcs.yolo_pipeline import YOLOPipeline

logger = get_logger(__name__)


class DLModelTrainerAgent:

    def run(self, state):
        logger.info("Running DLModelTrainerAgent")

        dataset_path = state.get("dataset_path")

        dataset_path_obj = Path(dataset_path)
        dataset_name = dataset_path_obj.name
        dataset_group = dataset_path_obj.parent.name

        artifacts_path = Path("yolo_train_artifacts") / dataset_group / f"{dataset_name}_roi640"

        pipeline = YOLOPipeline(
            raw_dataset=Path(dataset_path),
            processed_dataset=Path(dataset_path + "_roi640"),
            yolo_train_artifacts_save_path=artifacts_path,
            class_names=["object"]
        )
        pipeline.run()

        return state
