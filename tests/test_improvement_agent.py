
from src.agents.improvement_agent import improvement_agent

def test_improvement_agent():
    state = {
        "predictor_code": "abc",
        "failure_cases": [{"type": "FP"}],
        "logs": [],
    }

    print(state)
    result = improvement_agent(state)
    print(result["predictor_code"])

    assert "Improved" in result["predictor_code"]

if __name__ == "__main__":
    test_improvement_agent()
