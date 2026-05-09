
from src.utils.logger import log_step

def task_interpreter(state):
    state["dataset_info"] = {
        "task_type": "counting",
        "annotation_type": "bbox",
        "output_type": "scalar"
    }

    log_step(state, "Task interpreted")
    return state
