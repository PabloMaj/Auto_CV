
from src.agents.task_agent import task_agent

def test_task_agent():
    state = {
        "contracts": {},
        "logs": []
    }

    result = task_agent(state)

    assert result["contracts"]["metric"] == "MAE"
