
from src.agents.dataset_agent import dataset_agent

def test_dataset_agent():
    state = {
        "dataset_path": ".",
        "logs": []
    }

    result = dataset_agent(state)

    assert "dataset_report" in result
