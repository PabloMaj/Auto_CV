from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


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

    stage_id: Optional[int] = 1
    step_id: Optional[int] = 1
    generated_code: Optional[str] = None

    runner_success: Optional[bool] = None
    runner_output: Optional[str] = None
    runner_error: Optional[str] = None

    evaluation_metric: Optional[float] = None
    evaluation_summary: Optional[str] = None

    prediction_visualizations: List[str] = Field(default_factory=list)

    improvement_suggestions: List[str] = Field(default_factory=list)
    previous_solution_descriptions: List[str] = Field(default_factory=list)

    current_solution_iteration: int = 0
    current_improvement_step: int = 0

    previous_best_metric: float = -1.0

    logs: List[str] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)
