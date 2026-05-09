
from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict):
    task: str
    dataset_path: str
    annotations_path: str
    user_hints: str

    dataset_report: Dict[str, Any]
    contracts: Dict[str, Any]

    predictor_code: str
    evaluator_code: str

    metrics: Dict[str, Any]

    failures: List[Dict[str, Any]]

    iteration: int
    max_iterations: int

    experiment_memory: List[Dict[str, Any]]

    logs: List[str]

def initial_state():
    return {
        "task": "Count plants in RGB image",
        "dataset_path": "./data/images",
        "annotations_path": "./data/annotations",
        "user_hints": "Use classical CV if possible",

        "dataset_report": {},
        "contracts": {},

        "predictor_code": "",
        "evaluator_code": "",

        "metrics": {},

        "failures": [],

        "iteration": 0,
        "max_iterations": 3,

        "experiment_memory": [],

        "logs": []
    }
