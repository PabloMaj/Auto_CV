
from src.utils.logger import log_step

def test_logger():
    state = {"logs": []}

    state = log_step(state, "hello")

    assert len(state["logs"]) == 1
