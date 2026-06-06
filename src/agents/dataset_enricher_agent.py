from pathlib import Path

from src.funcs.dataset_enricher_funcs.yolo_sam_llm_pseudo_pipeline import YOLOSAMLLMPseudoPipeline
from src.utils.cuda import cuda_cleanup
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetEnricherAgent:

    def __init__(self, settings=None):
        self.settings = settings

    def run(self, state):
        logger.info("Running DatasetEnricherAgent")

        dataset_path = state.get("dl_dataset_path")

        dataset_path_obj = Path(dataset_path)
        dataset_name = dataset_path_obj.name
        dataset_group = dataset_path_obj.parent.name

        dataset_root = Path("data/data_structured") / dataset_group / dataset_name
        unlabeled_root = Path("data/data_structured") / dataset_group / dataset_name / "images" / "unlabelled"
        exp_id = state.get("exp_id", "default")
        output_root = Path("workspace") / exp_id / "dataset_enrichment_pseudo" / dataset_group / dataset_name

        sam_max_iters = self.settings.sam_prompt_optimizer_max_iters if self.settings else 5
        pseudo_threshold = self.settings.enricher_metric_threshold if self.settings else 0.0

        repo_root = Path(__file__).resolve().parents[2]
        sam_model_path = str(repo_root / "resources" / "sam3.pt")

        pipeline = YOLOSAMLLMPseudoPipeline(
            dataset_root=dataset_root, unlabeled_root=unlabeled_root, output_root=output_root,
            class_names=["object"], llm_model="gemma3:latest", task="detect", tile_size=640, overlap=0.5,
            sam_model_path=sam_model_path, prompt_optimizer_max_iters=sam_max_iters,
            pseudo_label_metric_threshold=pseudo_threshold,
        )
        pipeline.run()
        del pipeline
        cuda_cleanup("dataset_enricher")

        logger.info("Dataset enrichment completed")

        return state
