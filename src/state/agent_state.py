from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path


class EvalArtifact(BaseModel):
    step_key: str
    value: float
    img_paths: List[Path] = Field(default_factory=list)


class AgentState(BaseModel):
    user_prompt: str
    dl_dataset_path: str
    eval_dataset_path: Optional[str] = None
    exp_id: str = ""

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
    eval_artifacts: List[EvalArtifact] = Field(default_factory=list)

    runner_success: Optional[bool] = None
    runner_output: Optional[str] = None
    runner_error: Optional[str] = None
    total_retry_count: int = 0

    improvement_suggestions: List[str] = Field(default_factory=list)

    demo_app_exe_path: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
