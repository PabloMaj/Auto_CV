
from src.agents.runner_agent import runner_agent

def test_runner_agent():
    state = {
        "predictor_code": "print('hello')",
        "failures": [],
        "logs": []
    }

    result = runner_agent(state)

    assert result["runner_stderr"] == ""
