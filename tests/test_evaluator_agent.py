
from src.agents.evaluator_agent import evaluator_agent

def test_evaluator_agent():
    state = {
        "logs": []
    }

    result = evaluator_agent(state)

    assert result["evaluator_info"]["metric"] == "MAE"
