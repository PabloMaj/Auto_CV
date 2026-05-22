from pathlib import Path

from src.utils.logger import get_logger
from src.funcs.dl_model_trainer_funcs.yolo_pipeline import YOLOPipeline

logger = get_logger(__name__)


class DLModelTrainerAgent:

    def run(self, state):
        logger.info("Running DLModelTrainerAgent")

        dataset_path = state.get("dataset_path")

        pipeline = YOLOPipeline(
            raw_dataset=Path(dataset_path),
            processed_dataset=Path(dataset_path + "_roi640"),
            yolo_train_artifacts_save_path=Path(r"C:\projects\agent_cv\yolo_train_artifacts\crop_line_uav\maize_3_nerac_2016_1_roi640"),
            class_names=["crop"]
        )
        pipeline.run()

        return state
