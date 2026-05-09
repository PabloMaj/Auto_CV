
from src.agents.evaluation_agent import evaluation_agent

def test_evaluation_agent():
    state = {
        "metrics": {},
        "failures": [],
        "experiment_memory": [],
        "iteration": 0,
        "logs": []
    }

    result = evaluation_agent(state)

    assert "MAE" in result["metrics"]
