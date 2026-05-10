
from src.agents.runner import RunnerAgent


def test_runner_creation():
    agent = RunnerAgent()
    assert agent is not None
