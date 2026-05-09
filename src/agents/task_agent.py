
from src.logger import log

def task_agent(state):
    state["contracts"] = {
        "input": "RGB image",
        "output": "scalar_count",
        "metric": "MAE"
    }

    log(state, "Task formalized")
    return state
