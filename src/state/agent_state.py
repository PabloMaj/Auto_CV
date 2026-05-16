
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class AgentState(BaseModel):
    user_prompt: str
    dataset_path: str

    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    unlabeled_samples: int = 0

    task_type_for_dataset: Optional[str] = None
    enrichement_for_dataset_needed: Optional[bool] = None
    desired_output: Optional[str] = None

    generated_code: Optional[str] = None
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None

    evaluation_metric: Optional[float] = None
    evaluation_summary: Optional[str] = None

    prediction_visualizations: List[str] = []

    improvement_suggestions: List[str] = []
    previous_solution_descriptions: List[str] = []

    current_solution_iteration: int = 0
    current_improvement_step: int = 0

    previous_best_metric: float = -1.0

    logs: List[str] = []

    metadata: Dict[str, Any] = {}
