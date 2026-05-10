
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

        return state
