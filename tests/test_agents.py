
from src.agents.data_preprocessor import DataPreprocessorAgent


def test_preprocessor():
    state = {
        "dataset_path": "./dataset"
    }

    agent = DataPreprocessorAgent()
    result = agent.run(state)

    assert "train_samples" in result
