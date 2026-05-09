
from src.agents.programmer_agent import programmer_agent

def test_programmer_agent():
    state = {
        "iteration": 0,
        "logs": []
    }

    result = programmer_agent(state)

    assert "Predictor" in result["predictor_code"]
