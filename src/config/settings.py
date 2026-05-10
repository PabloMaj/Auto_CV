
from pydantic import BaseModel


class SystemSettings(BaseModel):
    enable_dl_model_trainer: bool = True
    enable_dataset_enricher: bool = True

    n_novel_solutions: int = 2
    m_improvement_steps: int = 3

    enable_iterative_improvement: bool = True
    enable_novel_solution_search: bool = True

    max_runner_retries: int = 3

    programmer_model: str = "qwen2.5-coder:7b"
    improvement_model: str = "qwen2.5vl:7b"
