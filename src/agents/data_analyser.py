
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataAnalyserAgent:

    def run(self, state):
        logger.info("Running DataAnalyserAgent")

        # TODO: inspect annotations
        # TODO: determine task type

        state["detected_task_type"] = "instance_segmentation"
        state["proposed_dl_model"] = "Mask R-CNN"

        return state
