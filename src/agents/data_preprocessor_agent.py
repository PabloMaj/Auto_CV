from src.utils.logger import get_logger
from src.funcs.data_preprocessor_funcs.data_preprocessor_funcs import get_split_paths, count_images, count_objects, get_vis_path

logger = get_logger(__name__)


class DataPreprocessorAgent:

    def run(self, state):
        logger.info("Running DataPreprocessorAgent")

        dataset_path = state.get("dataset_path")

        if not dataset_path:
            raise ValueError("dataset_path is missing")

        # =========================
        # PATHS
        # =========================
        split_paths = get_split_paths(dataset_path)
        state["split_paths"] = split_paths

        # =========================
        # IMAGE COUNTS
        # =========================
        image_counts = count_images(split_paths)
        state["image_counts"] = image_counts

        # =========================
        # OBJECT COUNTS
        # =========================
        object_counts = count_objects(split_paths)
        state["object_counts"] = object_counts

        # =========================
        # VIS PATH
        # =========================
        state["vis_path"] = str(get_vis_path(dataset_path))

        # =========================
        # FLATTENED STATS
        # =========================
        state["train_samples"] = image_counts.get("train", 0)
        state["val_samples"] = image_counts.get("val", 0)
        state["test_samples"] = image_counts.get("test", 0)
        state["unlabelled_samples"] = image_counts.get("unlabelled", 0)

        state["train_objects"] = object_counts.get("train", 0)
        state["val_objects"] = object_counts.get("val", 0)
        state["test_objects"] = object_counts.get("test", 0)
        state["unlabelled_objects"] = object_counts.get("unlabelled", 0)

        logger.info("DataPreprocessorAgent finished successfully")

        return state
