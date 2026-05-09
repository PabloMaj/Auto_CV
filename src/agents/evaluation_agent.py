
import random
from src.logger import log

def evaluation_agent(state):

    metric = round(random.uniform(0.1, 3.0), 3)

    state["metrics"] = {
        "MAE": metric
    }

    if metric > 1.0:
        state["failures"].append({
            "type": "high_error",
            "metric": metric
        })

    state["experiment_memory"].append({
        "iteration": state["iteration"],
        "metric": metric
    })

    log(state, f"Evaluation completed. MAE={metric}")

    return state
