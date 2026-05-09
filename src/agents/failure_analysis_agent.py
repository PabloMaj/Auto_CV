
from src.logger import log

def failure_analysis_agent(state):

    recommendations = []

    for failure in state["failures"]:

        if failure["type"] == "high_error":
            recommendations.append(
                "Improve thresholding"
            )

        if failure["type"] == "runtime_error":
            recommendations.append(
                "Fix runtime exception"
            )

    state["recommendations"] = recommendations

    log(state, "Failure analysis completed")

    return state
