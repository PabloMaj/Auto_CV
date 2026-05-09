
from src.agents.failure_analysis_agent import failure_analysis_agent

def test_failure_analysis():

    state = {
        "failures": [
            {
                "type": "high_error"
            }
        ],
        "logs": []
    }

    result = failure_analysis_agent(state)

    assert len(result["recommendations"]) > 0
