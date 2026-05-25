from pathlib import Path

from src.funcs.dataset_enricher_funcs.yolo_sam_llm_pseudo_pipeline import YOLOSAMLLMPseudoPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetEnricherAgent:

    def run(self, state):
        logger.info("Running DatasetEnricherAgent")

        dataset_path = state.get("dataset_path")

        dataset_path_obj = Path(dataset_path)
        dataset_name = dataset_path_obj.name
        dataset_group = dataset_path_obj.parent.name

        dataset_root = Path("data/data_structured") / dataset_group / dataset_name
        unlabeled_root = Path("data/data_structured") / dataset_group / dataset_name / "images" / "unlabelled"
        output_root = Path("dataset_enrichment_pseudo") / dataset_group / dataset_name

        pipeline = YOLOSAMLLMPseudoPipeline(dataset_root=dataset_root, unlabeled_root=unlabeled_root, output_root=output_root,
                                            class_names=["object"], llm_model="gemma3:latest", task="detect", tile_size=640, overlap=0.5)
        pipeline.run()

        logger.info("Dataset enrichment completed")

        return state
