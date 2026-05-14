
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessorAgent:

    def run(self, state):
        logger.info("Running DataPreprocessorAgent")

        # TODO: implementation needed
        # - scan dataset
        # - split train/val/test
        # - normalize data
        # - save metadata

        state["train_samples"] = 100
        state["val_samples"] = 20
        state["test_samples"] = 20
        state["unlabeled_samples"] = 50

        ROOT_DATA_DIR = "C:/projects/agent_cv/data/data_raw/maize/2_blois_2019_1/"
        state["path_to_train_images"] = [os.path.join(ROOT_DATA_DIR, f) for f in os.listdir(ROOT_DATA_DIR) if f.endswith((".jpg", ".png"))]

        return state
