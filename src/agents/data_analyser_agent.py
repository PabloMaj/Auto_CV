from pathlib import Path

from src.utils.logger import get_logger
from src.funcs.data_analyser_funcs.data_analyser_funcs import determine_desired_output, build_desired_output_definition
from src.inference.ollama_inference import OllamaInference

logger = get_logger(__name__)


class DataAnalyserAgent:

    def __init__(self, inference=OllamaInference(model="qwen2.5vl:7b")):
        self.inference = inference

    def run(self, state):
        repo_root = Path(__file__).resolve().parents[2]
        self.inference.log_dir = repo_root / "workspace" / state.get("exp_id", "default") / "llm_logs" / "data_analyser"

        logger.info("Running DataAnalyserAgent")

        # decision about task type for dataset
        state["task_type_for_dataset"] = "object_detection"

        # decision about whether dataset enrichment is needed
        if state["task_type_for_dataset"] in ["object_detection", "semantic_segmentation"]:
            state["enrichement_for_dataset_needed"] = True
        else:
            state["enrichement_for_dataset_needed"] = False

        # allowed output forms for CV tasks
        output_forms_allowed = ["scalar", "points", "line_segments", "bounding_boxes", "polygons", "segmentation_masks"]

        # extract user prompt from state
        user_prompt = state["user_prompt"]

        # decision about desired output format for CV task
        state["desired_output"] = determine_desired_output(self.inference, user_prompt, output_forms_allowed)
        state["desired_output_definition"] = build_desired_output_definition(state["desired_output"])

        return state
