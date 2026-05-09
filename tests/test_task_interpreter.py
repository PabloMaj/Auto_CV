
from src.agents.task_interpreter import task_interpreter

def test_task_interpreter():
    state = {
        "dataset_info": {},
        "logs": []
    }

    result = task_interpreter(state)

    assert result["dataset_info"]["task_type"] == "counting"
